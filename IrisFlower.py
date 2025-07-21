import argparse
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import BatchNormalization, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import Input

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import classification_report, confusion_matrix


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap


def load_and_preprocess_data():
    iris = load_iris()
    x = iris.data
    y = iris.target

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    y_encoded = to_categorical(y)
    
    return x_scaled, y_encoded, iris, scaler


def create_model(input_shape=(4,)):
    model = Sequential([
        Input(shape=input_shape),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(64, activation ='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation = 'relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(3, activation = 'softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model(model, x_train, y_train, epochs, batch_size, validation_split=0.2):
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
    checkpoint = ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)

    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_split=validation_split,
        callbacks=[early_stop, checkpoint, reduce_lr]
    )
    return history


def load_trained_model(path='best_model.h5'):
    print(f"Model {path} dosyasından yükleniyor...")
    model = load_model(path)
    return model


def evaluate_and_report(model, x_test, y_test, iris, scaler, x_scaled, y_true):
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test doğruluğu: {accuracy:.4f}")

    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    wrong_indices = np.where(y_pred_classes != y_true)[0]

    if len(wrong_indices) > 0:
        i = wrong_indices[0]
        print(f"Örnek index: {i}")
        print(f"Gerçek sınıf: {iris.target_names[y_true[i]]}")
        print(f"Tahmin edilen sınıf: {iris.target_names[y_pred_classes[i]]}")
        print(f"Özellikler (ölçeklenmiş): {x_test[i]}")

        print("Model tahmin olasılıkları:")
        for idx, prob in enumerate(y_pred[i]):
            print(f"  {iris.target_names[idx]}: {prob:.4f}")

        print("Özellikler (ölçeklenmemiş):")
        print(scaler.inverse_transform([x_test[i]])[0])

        plt.figure(figsize=(6,4))
        plt.scatter(x_test[:, 0], x_test[:, 1], c=y_true, cmap='viridis', alpha=0.5, label='Test verisi')
        plt.scatter(x_test[i, 0], x_test[i, 1], color='red', s=150, marker='X', label='Yanlış tahmin')
        plt.xlabel('Sepal Uzunluğu (ölçeklenmiş)')
        plt.ylabel('Sepal Genişliği (ölçeklenmiş)')
        plt.legend()
        plt.title('Yanlış Tahmin Edilen Örnek Konumu')
        plt.show()

    else:
        print("Model test setinde yanlış tahmin yapmadı.")

    print(classification_report(y_true, y_pred_classes, target_names=iris.target_names))
    print(f"Yanlış tahmin edilen örnek sayısı: {len(wrong_indices)}")

    for i in wrong_indices:
        print(f"Örnek index: {i}")
        print(f"Gerçek sınıf: {iris.target_names[y_true[i]]}")
        print(f"Tahmin edilen sınıf: {iris.target_names[y_pred_classes[i]]}")
        print(f"Özellikler (ölçeklenmiş): {x_test[i]}")
        print("------")

    plt.figure(figsize=(8,6))
    correct_indices = np.where(y_pred_classes == y_true)[0]

    plt.scatter(x_test[correct_indices, 0], x_test[correct_indices, 1], 
                c='green', label='Doğru Tahmin', alpha=0.6)
    plt.scatter(x_test[wrong_indices, 0], x_test[wrong_indices, 1], 
                c='red', label='Yanlış Tahmin', marker='x', s=100)

    plt.xlabel('Özellik 1 (ölçeklenmiş)')
    plt.ylabel('Özellik 2 (ölçeklenmiş)')
    plt.legend()
    plt.title('Doğru ve Yanlış Tahminler (iki özellik üzerinde)')
    plt.show()

    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=iris.target_names,
                yticklabels=iris.target_names)
    plt.xlabel('Tahmin')
    plt.ylabel('Gerçek')
    plt.title('Confusion Matrix')
    plt.show()


def plot_training_history(history):
    plt.plot(history.history['loss'], label='Eğitim kaybı')
    plt.plot(history.history['val_loss'], label='Doğrulama kaybı')
    plt.title('Model Kayıp Değerleri')
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp')
    plt.legend()
    plt.show()

    plt.plot(history.history['accuracy'], label='Eğitim doğruluğu')
    plt.plot(history.history['val_accuracy'], label='Doğrulama doğruluğu')
    plt.title('Model Doğruluk Değerleri')
    plt.xlabel('Epoch')
    plt.ylabel('Doğruluk')
    plt.legend()
    plt.show()


def shap_explain(model, x_train, x_test, y_pred_classes, iris):
    background = x_train[np.random.choice(x_train.shape[0], 50, replace=False)]
    pred_class = y_pred_classes[0]

    def f(x):
        return model.predict(x)[:, pred_class]

    explainer = shap.KernelExplainer(f, background)
    shap_values = explainer.shap_values(x_test[:10])

    shap.initjs()
    shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        x_test[0],
        feature_names=iris.feature_names
    )



def kfold_cross_validation(x_scaled, y_encoded, iris, k=5, epochs=100, batch_size=5):
    y_integers = np.argmax(y_encoded, axis=1)  # One-hot'dan sınıf integer'larına dönüşüm
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    accuracies = []

    for fold, (train_index, val_index) in enumerate(skf.split(x_scaled, y_integers), 1):
        x_train_cv, x_val_cv = x_scaled[train_index], x_scaled[val_index]
        y_train_cv, y_val_cv = y_encoded[train_index], y_encoded[val_index]

        model_cv = create_model(input_shape=(x_scaled.shape[1],))

        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        history = model_cv.fit(
            x_train_cv, y_train_cv,
            validation_data=(x_val_cv, y_val_cv),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[early_stop]
        )

        loss, accuracy = model_cv.evaluate(x_val_cv, y_val_cv, verbose=0)
        accuracies.append(accuracy)

        y_val_pred_prob = model_cv.predict(x_val_cv)
        y_val_pred = np.argmax(y_val_pred_prob, axis=1)
        y_val_true = np.argmax(y_val_cv, axis=1)

        print(f"\nFold {fold} - Doğruluk: {accuracy:.4f}")
        print(classification_report(y_val_true, y_val_pred, target_names=iris.target_names, zero_division=0))

        cm = confusion_matrix(y_val_true, y_val_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=iris.target_names,
                    yticklabels=iris.target_names)
        plt.title(f'Confusion Matrix - Fold {fold}')
        plt.xlabel('Tahmin')
        plt.ylabel('Gerçek')
        plt.show()

    print(f'\nOrtalama doğruluk: {np.mean(accuracies):.4f}')
    print(f'Standart sapma: {np.std(accuracies):.4f}')




def main():
    global args
    parser = argparse.ArgumentParser(description="Iris Veri Seti ile Keras Model Eğitimi")
    parser.add_argument('--epochs', type=int, default=100, help='Eğitim epoch sayısı')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch boyutu')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test veri oranı')
    parser.add_argument('--train_model', action='store_true', help='Modeli eğit')
    parser.add_argument('--use_saved_model', action='store_true', help='Kaydedilmiş modeli kullan')
    args = parser.parse_args()

    x_scaled, y_encoded, iris, scaler = load_and_preprocess_data()
    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled, y_encoded, test_size=args.test_size, random_state=42
    )
    
    if args.use_saved_model:
        model = load_trained_model()
    else:
        model = create_model()
        if args.train_model:
            history = train_model(model, x_train, y_train, epochs=args.epochs, batch_size=args.batch_size)
            plot_training_history(history)
        else:
            print("Model eğitilmedi, yeni bir model oluşturuldu.")

    evaluate_and_report(model, x_test, y_test, iris, scaler, x_scaled, np.argmax(y_test, axis=1))

    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    shap_explain(model, x_train, x_test, y_pred_classes, iris)
    kfold_cross_validation(x_scaled, y_encoded, iris, k=5)


if __name__ == '__main__':
    main()
