import os
import json
from flask import Flask, request, redirect, url_for, render_template, jsonify, send_file
from google.cloud import storage
import google.generativeai as google_ai
from werkzeug.utils import secure_filename
from PIL import Image
import io

app = Flask(__name__)

app.jinja_env.filters['from_json'] = json.loads

GCS_BUCKET_NAME = 'cnd2bucket'
storage_client = storage.Client()
bucket = storage_client.bucket(GCS_BUCKET_NAME)

API_KEY = os.getenv('GOOGLE_AI_API_KEY')
if API_KEY:
    google_ai.configure(api_key=API_KEY)
else:
    print("API key is missing. Please set the GOOGLE_AI_API_KEY environment variable.")
model = google_ai.GenerativeModel('gemini-1.5-flash')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def call_google_gemini_ai(prompt, image_bytes):
    try:
        response = model.generate_content([prompt, image_bytes])
        return response.text
    except Exception as e:
        print(f"Error calling Google Generative AI: {str(e)[:100]}")
        return None

def clean_and_parse_json(text):
    if not text:
        print("Empty AI response.")
        return None
    text = text.strip("`json").strip("`")
    try:
        json_start = text.find('{')
        json_end = text.rfind('}') + 1
        if json_start == -1 or json_end == -1:
            print("No JSON found in AI response.")
            return None
        json_text = text[json_start:json_end]
        return json.loads(json_text)
    except json.JSONDecodeError as error:
        print(f"Error parsing JSON: {error}")
        return None

class Namespace:
    pass

@app.route('/')
def index():
    blobs = bucket.list_blobs()
    # Keep the original URLs to extract filenames later
    image_urls = [blob.public_url for blob in blobs if not blob.name.endswith('.json')]
    namespace = Namespace()
    namespace.bucket = bucket
    return render_template('index.html', images=image_urls, namespace=namespace)

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"}), 400
    image = request.files['image']
    if image.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(image.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(image.filename)
    blob = bucket.blob(filename)

    # Resize the image before uploading
    image_data = image.read()
    pil_image = Image.open(io.BytesIO(image_data))
    resized_image = resize_image(pil_image, (800, 600))  # Resize to max width 800px, height 600px
    image_bytes = io.BytesIO()
    resized_image.save(image_bytes, format=pil_image.format)
    image_bytes.seek(0)
    blob.upload_from_file(image_bytes, content_type=image.content_type)

    # Analyze the image using AI
    prompt = """
    Analyze the uploaded image and respond in the following JSON format:
    {
        "description": "A concise description of the image",
        "caption": "A short caption for the image"
    }
    """

    ai_response = call_google_gemini_ai(prompt, resized_image)

    if not ai_response:
        return jsonify({"error": "AI did not return a response"}), 500
    parsed_response = clean_and_parse_json(ai_response)
    if not parsed_response:
        return jsonify({"error": "Failed to parse AI response"}), 500

    metadata = {
        "description": parsed_response.get("description", "No description available"),
        "caption": parsed_response.get("caption", "No caption available")
    }

    json_blob = bucket.blob(f"{os.path.splitext(filename)[0]}.json")
    json_blob.upload_from_string(json.dumps(metadata), content_type='application/json')

    return redirect(url_for('index'))

# New route to serve images
@app.route('/image/<filename>')
def get_image(filename):
    blob = bucket.blob(filename)
    image_data = blob.download_as_bytes()

    # Optionally, resize the image on-the-fly when serving
    pil_image = Image.open(io.BytesIO(image_data))
    resized_image = resize_image(pil_image, (800, 600))  # Resize to max width 800px, height 600px
    output = io.BytesIO()
    resized_image.save(output, format=pil_image.format)
    output.seek(0)

    return send_file(output, mimetype=blob.content_type)

def resize_image(image, size):
    """
    Resize an image while maintaining its aspect ratio.
    :param image: PIL.Image object
    :param size: Tuple (max_width, max_height)
    :return: Resized PIL.Image object
    """
    # Use Image.Resampling.LANCZOS for high-quality resizing
    image.thumbnail(size, Image.Resampling.LANCZOS)
    return image

if __name__ == '__main__':
    app.run(debug=True)