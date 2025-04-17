from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

def build_transfer_model(input_shape=(48, 48, 3), num_classes=7):
    """Xây dựng mô hình Transfer Learning với VGG16."""
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Đóng băng để huấn luyện ban đầu
    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model