# 🌦️ Analisis Cuaca & Curah Hujan Harian di Indonesia (2015–2020)

Proyek ini merupakan implementasi analisis data iklim harian di Indonesia dari tahun 2015 hingga 2020.  
Data diolah untuk memahami tren suhu dan curah hujan, kemudian digunakan dalam model machine learning untuk melakukan prediksi dan segmentasi cuaca.  
Seluruh hasil ditampilkan secara interaktif melalui dashboard berbasis **Streamlit**.

---

## 🎯 Tujuan Proyek

- Menganalisis pola musiman suhu dan curah hujan
- Mengelompokkan hari berdasarkan kondisi cuaca dengan **K-Means Clustering**
- Memprediksi suhu rata-rata harian menggunakan **regresi linear**
- Menyajikan hasil secara interaktif menggunakan **dashboard Python**

---

## 📁 Struktur Proyek

| File / Folder | Deskripsi |
|---------------|-----------|
| `FP-BDPA-FINAL.ipynb` | Notebook utama untuk EDA, modeling, dan visualisasi |
| `app.py` | Dashboard interaktif dengan Streamlit |
| `Indonesia daily climate data from 2010 to 2022 dataset` | Dataset mentah utama (besar) |
| `climate_day_2015_2020.csv` | Dataset hasil preprocessing |
| `README.md` | Dokumentasi proyek ini |

---

## ⚙️ Teknologi & Library

- Python 3.x
- Pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn
- Streamlit

---

## ▶️ Cara Menjalankan Proyek

Untuk menjalankan proyek ini di komputer lokal kamu, ikuti langkah-langkah berikut:

1. Clone repository  
   Jalankan perintah:  
   `git clone https://github.com/username/Analisis-Cuaca-Curah-Hujan-Harian-di-Indonesia-2015-2020.git`  
   `cd Analisis-Cuaca-Curah-Hujan-Harian-di-Indonesia-2015-2020`

2. Install dependencies  
   Pastikan Python versi 3.7 atau lebih baru telah terpasang.  
   Install library yang dibutuhkan dengan perintah:  
   `pip install -r requirements.txt`  
   Jika belum memiliki Streamlit:  
   `pip install streamlit`

3. Jalankan Dashboard Streamlit  
   Jalankan perintah:  
   `streamlit run app.py`

