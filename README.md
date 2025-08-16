# 🌸 Iris Classification with Keras & SHAP

Bu proje, **Iris veri seti** üzerinde bir **derin öğrenme modeli** (Keras ile) eğitilmesi, değerlendirilmesi ve yorumlanmasını içerir.  
Model, çiçeklerin özelliklerine (sepal length, sepal width, petal length, petal width) bakarak üç sınıftan birini (Setosa, Versicolor, Virginica) tahmin eder.  

## 🚀 Özellikler
- **Tam veri ön işleme** (StandardScaler ile normalizasyon, one-hot encoding).
- **Keras Sequential Model** ile derin sinir ağı:
  - Dense katmanlar, BatchNormalization ve Dropout kullanımı.
- **Erken durdurma, öğrenme oranı azaltma ve model checkpoint** callback’leri.
- **Eğitim geçmişi grafikleri** (loss ve accuracy).
- **Confusion Matrix** görselleştirmesi.
- **Yanlış tahmin edilen örneklerin analizi**.
- **SHAP (Explainable AI)** ile model yorumlama.
- **K-fold cross validation** (StratifiedKFold ile).

## 📦 Gereksinimler
Aşağıdaki kütüphaneler yüklü olmalıdır:
```bash
pip install tensorflow scikit-learn matplotlib seaborn shap

## 📂 Proje Yapısı
├── main.py              # Ana script (model eğitimi, değerlendirme, SHAP, K-Fold)
├── best_model.h5        # Kaydedilmiş en iyi model (eğitim sonrası oluşur)
├── README.md            # Proje dökümantasyonu
