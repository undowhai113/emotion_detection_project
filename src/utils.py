import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def load_fer2013(data_path):
    """Đọc file CSV và trả về dữ liệu ảnh và nhãn."""
    data = pd.read_csv(data_path)
    pixels = data['pixels'].apply(lambda x: np.array(x.split(), dtype='float32'))
    images = np.array([x.reshape(48, 48, 1) for x in pixels])
    labels = data['emotion'].values
    return images, labels

def preprocess_data(images, labels, test_size=0.2, val_size=0.1):
    """Chuẩn hóa dữ liệu và chia thành tập train, val, test."""
    images = images / 255.0
    labels = to_categorical(labels, num_classes=7)
    X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=test_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size / test_size)
    return X_train, y_train, X_val, y_val, X_test, y_test

def save_processed_data(X_train, y_train, X_val, y_val, X_test, y_test, save_dir):
    """Lưu dữ liệu đã tiền xử lý thành file .npy."""
    np.save(f"{save_dir}/X_train.npy", X_train)
    np.save(f"{save_dir}/y_train.npy", y_train)
    np.save(f"{save_dir}/X_val.npy", X_val)
    np.save(f"{save_dir}/y_val.npy", y_val)
    np.save(f"{save_dir}/X_test.npy", X_test)
    np.save(f"{save_dir}/y_test.npy", y_test)