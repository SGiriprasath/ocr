import os
import os
import sqlite3
import boto3
import cv2
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, render_template, jsonify, send_file
from werkzeug.utils import secure_filename
import base64
import requests
import zipfile
from io import BytesIO
import logging

app = Flask(__name__)

# Configure AWS credentials
os.environ['AWS_ACCESS_KEY_ID'] = ''
os.environ['AWS_SECRET_ACCESS_KEY'] = ''
os.environ['AWS_REGION'] = ''
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
region_name = os.getenv('AWS_REGION')
aws_bucket = 'photofilterocr'
table_name_1 = aws_bucket.replace("-","") + "_table_1"
table_name_2 = aws_bucket.replace("-","") + "_table_2"

s3 = boto3.client(
    's3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=region_name
)

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, max_batch_size=200)


def already_exist_data():
    conn = sqlite3.connect('my_database.db')
    c = conn.cursor()
    already_exist = []
    db_path = "my_database.db"  # Replace with your database file path
    table_name_to_check = table_name_1
    try:
        conn = sqlite3.connect(db_path)
        if table_exists(conn, table_name_to_check):
            print(f"The table '{table_name_to_check}' exists.")
            exist_data = c.execute(f"SELECT image_name FROM {table_name_1}").fetchall()
            for data in exist_data:
                already_exist.append(data[0])
        else:
            print(f"The table '{table_name_to_check}' does not exist.")
    except sqlite3.Error as e:
        print(f"Error: {e}")
    finally:
        conn.close()

    return already_exist


def table_exists(conn, table_name):
    query = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';"
    cursor = conn.execute(query)
    return cursor.fetchone() is not None


def remove_consecutive_duplicates(nums):
    result = []
    prev_num = None
    for num in nums:
        if num != prev_num:
            result.append(num)
        prev_num = num
    return result


def approximate_match(input_text, data_list):
    approximate_match_list = []
    for data, data2 in data_list:
        if input_text in data2 and input_text != data2:
            approximate_match_list.append(data)
    return remove_consecutive_duplicates(approximate_match_list)


def crop_and_zoom(img, x, y, zoom_factor):
    img1 = img.resize((8000, 8000), Image.LANCZOS)
    width, height = img1.size
    zoom_ratio = zoom_factor * 2
    cropped_img = img1.crop(
        (x - width / zoom_ratio, y - height / zoom_ratio, x + width / zoom_ratio, y + height / zoom_ratio))
    return cropped_img.resize((width, height), Image.LANCZOS)


def integrate_functions(img_pil, x, y, zoom_factor):
    cropped_zoomed_img_pil = crop_and_zoom(img_pil, x, y, zoom_factor)
    cropped_zoomed_img_cv = cv2.cvtColor(np.array(cropped_zoomed_img_pil), cv2.COLOR_RGB2BGR)
    return cropped_zoomed_img_cv


def fetch_image_from_s3(bucket_name, object_key):
    response = s3.get_object(Bucket=bucket_name, Key=object_key)
    image_data = response['Body'].read()
    img_pil = Image.open(BytesIO(image_data))
    return img_pil


def list_images_from_s3(bucket_name):
    objects = s3.list_objects_v2(Bucket=bucket_name)
    return [obj['Key'] for obj in objects.get('Contents', [])]


def ocr_with_preprocessing_from_s3(bucket_name, x, y, zoom_factor):
    image_dict = {}

    already_exist_data_value = already_exist_data()
    image_keys = list_images_from_s3(bucket_name)  # Get all image keys from the bucket

    def process_image(object_key):
        if object_key not in already_exist_data_value:
            img_pil = fetch_image_from_s3(bucket_name, object_key)  # Fetch image from S3
            preprocessed_img = integrate_functions(img_pil, x, y, zoom_factor)  # Apply preprocessing
            try:
                # Perform OCR on the preprocessed image
                result = ocr.ocr(preprocessed_img)
                print(f"OCR result for {object_key}: {result}")
                if result and result[0]:
                    img_names = [i[1][0].lower() if i[1][0].isalpha() else i[1][0] for data in result for i in data]
                    return object_key, img_names
            except Exception as e:
                logging.error(f"Error processing image {object_key}: {e}")
                return None
        return None

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=5) as executor:  # Adjust max_workers based on your CPU/GPU
        futures = [executor.submit(process_image, key) for key in image_keys]
        for future in as_completed(futures):
            res = future.result()
            if res:
                object_key, img_names = res
                image_dict[object_key] = img_names
                print("image:", image_dict)

    return image_dict


def initialize_database_with_text(image_text_dict):
    conn = sqlite3.connect('my_database.db')
    c = conn.cursor()

    # Create tables if not exist
    c.execute(
        f"CREATE TABLE IF NOT EXISTS {table_name_1} (id INTEGER PRIMARY KEY AUTOINCREMENT, image_name TEXT);")
    c.execute(
        f"CREATE TABLE IF NOT EXISTS {table_name_2} (id INTEGER PRIMARY KEY, image_id TEXT NOT NULL, image_text TEXT);")

    # Insert image names into images table
    for image_name in image_text_dict.keys():
        c.execute(f"INSERT INTO {table_name_1} (image_name) VALUES (?)", (image_name,))

    # Insert image text data into image_data table
    for image_name, image_text_list in image_text_dict.items():
        for image_text in image_text_list:
            c.execute(f"INSERT INTO {table_name_2} (image_id, image_text) VALUES (?, ?)", (image_name, image_text))
    conn.commit()
    conn.close()


def train_model():
    bucket_name = aws_bucket
    x, y, zoom_factor = 4000, 5000, 1  # Example values for cropping and zooming
    # Perform OCR with preprocessing for all images in the S3 bucket
    image_text_dict = ocr_with_preprocessing_from_s3(bucket_name, x, y, zoom_factor)
    initialize_database_with_text(image_text_dict)


def create_presigned_url(bucket_name, object_key, expiration=10000):
    try:
        response = s3.generate_presigned_url('get_object',
                                             Params={'Bucket': bucket_name,
                                                     'Key': object_key},
                                             ExpiresIn=expiration)
    except Exception as e:
        logging.error(f"Error generating presigned URL: {e}")
        return None
    return response


def search_images_by_text(input_text):
    image_list = []
    conn = sqlite3.connect("my_database.db")
    cursor = conn.cursor()
    approximate_match_list = cursor.execute(
        f"SELECT image_id, image_text FROM {table_name_2}").fetchall()
    list_data = cursor.execute(
        f"SELECT {table_name_2}.image_id FROM {table_name_2} WHERE {table_name_2}.image_text = ?",
        (input_text,)
    ).fetchall()
    approximate_match_data = approximate_match(input_text, approximate_match_list)
    for data in list_data:
        image_list.append(data[0])
    conn.close()
    return image_list + list(approximate_match_data)


@app.route('/')
def home():
    return render_template('index.html', message="Enter a BIB Number to find your image")


@app.route('/upload')
def upload():
    return render_template('upload.html')


@app.route('/index1')
def download_delete():
    return render_template('index1.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files' not in request.files:
        return "No file part"
    files = request.files.getlist('files')
    for file in files:
        if file.filename == '':
            continue
        filename = secure_filename(file.filename)
        s3.upload_fileobj(
            file,
            aws_bucket,
            filename,
            ExtraArgs={"ContentType": file.content_type}
        )
    return render_template('upload.html')


@app.route('/train', methods=['POST'])
def train():
    train_model()
    return jsonify({'status': 'completed'})


@app.route('/search', methods=['POST'])
def search():
    conn = sqlite3.connect('my_database.db')
    input_text = request.form.get('bib_number')
    template_to_render = request.form.get('template',
                                          'index1.html')  # Get the template choice from the form, default to index1.html

    if table_exists(conn, table_name_1):
        if input_text:
            if input_text.isalpha():
                input_text = input_text.lower()
            image_ids = search_images_by_text(input_text)
            image_paths = [create_presigned_url(aws_bucket, key) for key in image_ids if
                           create_presigned_url(aws_bucket, key)]
            message = "" if image_ids else "No image found"
        else:
            image_paths = []
            message = "Please enter a BIB number"
    else:
        message = "Please upload the images!!"
        image_paths = []

    return render_template(template_to_render, images=image_paths, message=message, input_text=input_text)


@app.route('/download', methods=['POST'])
def download_images():
    image_urls = request.json['urls']
    as_zip = request.json.get('as_zip', False)
    if not as_zip:
        # For individual files, encode each image in base64 and return as JSON
        image_data = []
        for idx, url in enumerate(image_urls):
            response = requests.get(url)
            image = BytesIO(response.content)
            image.seek(0)
            image_data.append(base64.b64encode(image.getvalue()).decode('utf-8'))
        return jsonify({'status': 'success', 'images': image_data})
    else:
        # Create a zip file
        memory_file = BytesIO()
        with zipfile.ZipFile(memory_file, 'w') as zf:
            for idx, url in enumerate(image_urls):
                response = requests.get(url)
                image = BytesIO(response.content)
                zf.writestr(f'image_{idx + 1}.png', image.getvalue())
        memory_file.seek(0)
        return send_file(memory_file, as_attachment=True, download_name='images.zip')


@app.route('/delete', methods=['POST'])
def delete_image():
    try:
        data = request.get_json()
        urls = data['urls']
        conn = sqlite3.connect('my_database.db')
        c = conn.cursor()
        for url in urls:
            key = url.split('/')[-1].split('?')[0]  # Assuming the URL ends with the key of the S3 object
            print(key)
            c.execute(f"DELETE FROM {table_name_2} WHERE image_id = ?", (key,))
            s3.delete_object(Bucket=aws_bucket, Key=key)
        conn.commit()
        conn.close()
        return jsonify({'message': 'Images deleted successfully'}), 200

    except NoCredentialsError:
        return jsonify({'message': 'Credentials not available'}), 403
    except PartialCredentialsError:
        return jsonify({'message': 'Incomplete credentials'}), 403
    except Exception as e:
        return jsonify({'message': str(e)}), 500


@app.route('/delete_all', methods=['POST'])
def delete_all_images():
    try:
        # List all objects in the bucket
        objects = s3.list_objects_v2(Bucket=aws_bucket)
        conn = sqlite3.connect('my_database.db')
        if 'Contents' in objects:
            for obj in objects['Contents']:
                s3.delete_object(Bucket=aws_bucket, Key=obj['Key'])
                conn.execute(f"DELETE FROM {table_name_2}")
                conn.execute(f"DELETE FROM {table_name_1}")
        conn.commit()
        conn.close()
        return jsonify({'message': 'All images deleted successfully'}), 200

    except NoCredentialsError:
        return jsonify({'message': 'Credentials not available'}), 403
    except PartialCredentialsError:
        return jsonify({'message': 'Incomplete credentials'}), 403
    except Exception as e:
        return jsonify({'message': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
