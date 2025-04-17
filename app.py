from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image

# Khởi tạo Flask app
app = Flask(__name__)

# Load mô hình đã huấn luyện (đảm bảo đường dẫn chính xác)
MODEL_PATH = 'models/transfer_learning_model.h5'
model = load_model(MODEL_PATH)

# Các nhãn cảm xúc (cần phù hợp với thứ tự của one-hot encoding)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Các thông điệp an ủi tương ứng
comfort_messages = {
    'Angry': "Hãy bình tĩnh nhé! Cố gắng hít sâu và thư giãn.",
    'Disgust': "Mọi thứ rồi sẽ ổn, hãy tin tưởng vào chính mình.",
    'Fear': "Không có gì đáng sợ cả, bạn mạnh mẽ lắm!",
    'Happy': "Thật tuyệt khi thấy nụ cười của bạn!",
    'Sad': "Mình ở đây bên bạn, mọi chuyện sẽ qua.",
    'Surprise': "Ôi, có vẻ là bạn đang rất ngạc nhiên!",
    'Neutral': "Một ngày yên bình, hãy tận hưởng sự bình yên này."
}

def preprocess_image(image_data):
    """
    Chuyển đổi chuỗi base64 thành ảnh, resize về (48, 48) và normalize.
    Giả sử dữ liệu ảnh gửi lên từ web là một chuỗi base64 của ảnh JPEG.
    """
    # Loại bỏ phần header ("data:image/jpeg;base64,...")
    header, encoded = image_data.split(',', 1)
    decoded = base64.b64decode(encoded)
    image = Image.open(BytesIO(decoded)).convert('RGB')
    # Resize ảnh về kích thước mà model của bạn mong đợi: (48, 48)
    image = image.resize((48, 48))
    img_array = np.array(image, dtype="float32") / 255.0
    # Thêm dimension batch
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.route('/')
def index():
    # Trang giao diện
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    image_data = data.get('image')  # Lấy chuỗi base64 từ dữ liệu JSON gửi lên

    # Tiền xử lý ảnh
    img = preprocess_image(image_data)

    # Dự đoán cảm xúc
    preds = model.predict(img)
    emotion_idx = np.argmax(preds)
    emotion = emotion_labels[emotion_idx]
    message = comfort_messages[emotion]

    # Trả về kết quả dạng JSON
    return jsonify({'emotion': emotion, 'message': message})

if __name__ == '__main__':
    app.run(debug=True)
