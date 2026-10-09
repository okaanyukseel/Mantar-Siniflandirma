# Mushroom Classification  
## SVM, KNN ve Random Forest ile Zehirli Mantar Tespiti

Bu proje, **mushroom** veri seti kullanılarak mantarların **zehirli (poisonous)** veya **yenilebilir (edible)** olup olmadığının makine öğrenmesi algoritmaları ile sınıflandırılmasını amaçlamaktadır.

Projede üç farklı model eğitilmiş ve karşılaştırılmıştır:
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Random Forest

---

## 🎯 Projenin Amacı

- Kategorik verilerle sınıflandırma pratiği yapmak  
- Farklı makine öğrenmesi modellerinin performanslarını karşılaştırmak  
- Confusion Matrix, ROC Curve ve Accuracy gibi metrikleri analiz etmek  

---

## 📂 Veri Seti

Veri seti depoda `mushrooms.csv` adıyla bulunmaktadır.

- **8124** satır, **23** sütun
- Hedef sütun: `class` → `p` (zehirli) / `e` (yenilebilir)
  - `e`: 4208 örnek, `p`: 3916 örnek
- Diğer 22 sütunun tamamı kategorik özelliklerdir (`cap-shape`, `cap-surface`, `cap-color`, `bruises`, `odor`, `gill-*`, `stalk-*`, `veil-*`, `ring-*`, `spore-print-color`, `population`, `habitat`).

Kodda hedef değişken `p = 1` (zehirli), `e = 0` (yenilebilir) olarak kodlanır.

---

## ⚙️ Yöntem

1. `mushrooms.csv` okunur (dosya bulunamazsa `FileNotFoundError` verilir).
2. Tüm özellikler `OneHotEncoder` ile one-hot kodlanır.
3. Veri, `stratify` kullanılarak %80 eğitim / %20 test olarak ayrılır (`random_state=42`).
4. Üç model `Pipeline` içinde eğitilir:

| Model | Ön işleme | Parametreler |
|---|---|---|
| SVM | OneHot + `StandardScaler(with_mean=False)` | `kernel="rbf"`, `C=1.0`, `gamma="scale"`, `probability=True` |
| KNN | OneHot + `StandardScaler(with_mean=False)` | `n_neighbors=5`, `weights="distance"` |
| Random Forest | Yalnızca OneHot | `n_estimators=300`, `max_depth=8`, `min_samples_split=5`, `min_samples_leaf=3`, `max_features="sqrt"` |

5. Her model için test verisinde:
   - Sınıflandırma raporu (precision, recall, f1-score) yazdırılır
   - Confusion Matrix çizdirilir
   - ROC eğrisi çizdirilir
6. Son olarak modeller doğruluk (accuracy) değerine göre sıralanarak bir tablo halinde yazdırılır.

---

## 📊 Sonuçlar

Depoda kaydedilmiş bir çıktı bulunmamaktadır. Sonuçları (sınıflandırma raporları, confusion matrix ve ROC grafikleri, sıralı accuracy tablosu) görmek için betiği çalıştırın.

---

## 🗂️ Proje Yapısı

```
.
├── # Mantar Sınıflandırması.py   # Ana betik (veri hazırlama, model eğitimi, değerlendirme)
├── mushrooms.csv                 # Veri seti
├── requirements.txt
└── README.md
```

---

## ▶️ Çalıştırma

```bash
pip install -r requirements.txt
python "# Mantar Sınıflandırması.py"
```

Betik, `mushrooms.csv` dosyasını çalışma dizininde arar; bu nedenle komutu depo klasörü içinden çalıştırın. Grafikler `plt.show()` ile ayrı pencerelerde açılır.

---

## 📦 Gereksinimler

- Python 3
- pandas
- numpy
- matplotlib
- scikit-learn (`OneHotEncoder(sparse_output=...)` parametresi kullanıldığı için 1.2 veya üzeri bir sürüm gerekir)
