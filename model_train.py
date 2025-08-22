import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from sklearn.model_selection import train_test_split

# Veri setini yükle (gerçek veriyle değiştirilmesi gerekir)
# Bu örnekte sentetik veri kullanılıyor
def generate_synthetic_data(samples=10000):
    # Normal manyetik alan verileri (gürültülü sinüzoidal)
    t = np.linspace(0, 20*np.pi, samples)
    normal_data = np.sin(t) + 0.1 * np.random.randn(samples)
    
    # Anomali verileri (ani değişimler)
    anomaly_data = normal_data.copy()
    
    # Rastgele anomali noktaları ekle
    anomaly_indices = np.random.choice(samples, size=int(samples*0.05), replace=False)
    anomaly_data[anomaly_indices] += np.random.choice([-2, 2], size=len(anomaly_indices))
    
    # Etiketleri oluştur (0: normal, 1: anomali)
    labels = np.zeros(samples)
    labels[anomaly_indices] = 1
    
    return normal_data, anomaly_data, labels

# Veri setini hazırla
def prepare_dataset(sequences, labels, window_size=10):
    X, y = [], []
    for i in range(len(sequences) - window_size):
        X.append(sequences[i:i+window_size])
        y.append(labels[i+window_size])
    return np.array(X), np.array(y)

# Modeli oluştur
def create_model(input_shape):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(16, activation='relu'),
        layers.Dense(8, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Ana eğitim fonksiyonu
def main():
    # Sentetik veri oluştur
    print("Sentetik veri oluşturuluyor...")
    normal_data, anomaly_data, labels = generate_synthetic_data(10000)
    
    # Veri setini hazırla
    print("Veri seti hazırlanıyor...")
    X, y = prepare_dataset(anomaly_data, labels)
    
    # Eğitim ve test olarak ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Modeli oluştur
    print("Model oluşturuluyor...")
    model = create_model((X_train.shape[1],))
    
    # Modeli eğit
    print("Model eğitiliyor...")
    history = model.fit(X_train, y_train,
                        epochs=20,
                        batch_size=32,
                        validation_split=0.1,
                        verbose=1)
    
    # Modeli değerlendir
    print("Model değerlendiriliyor...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test doğruluğu: {test_acc:.4f}")
    
    # Modeli TensorFlow Lite formatına dönüştür
    print("TensorFlow Lite modeline dönüştürülüyor...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    # Modeli dosyaya kaydet
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print("Model kaydedildi: model.tflite")
    
    # Modeli hexadecimal formatında da kaydet (Arduino'da kullanmak için)
    hex_array = []
    for byte in tflite_model:
        hex_array.append(f"0x{byte:02x}")
    
    with open('model.h', 'w') as f:
        f.write("const unsigned char g_model[] = {\n")
        for i in range(0, len(hex_array), 12):
            f.write("  " + ", ".join(hex_array[i:i+12]) + ",\n")
        f.write("};\n")
        f.write(f"const int g_model_len = {len(tflite_model)};")
    
    print("Model header dosyası kaydedildi: model.h")

if __name__ == "__main__":
    main()
