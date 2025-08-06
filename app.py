import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("climate_data.csv")
    df['date'] = pd.to_datetime(df['date'], format="%d-%m-%Y")
    df = df[(df['date'].dt.year >= 2015) & (df['date'].dt.year <= 2020)]
    df = df.dropna()
    df_grouped = df.groupby('date').agg({
        'Tn': 'mean',
        'Tx': 'mean',
        'Tavg': 'mean',
        'RH_avg': 'mean',
        'RR': 'mean',
        'ss': 'mean',
        'ff_x': 'mean',
        'ddd_x': 'mean',
        'ff_avg': 'mean',
        'station_id': 'first',
        'ddd_car': lambda x: x.mode().iloc[0] if not x.mode().empty else None
    }).reset_index()
    return df_grouped

df = load_data()
st.title("Dashboard Analisis Data Iklim 2015-2020")

# Sidebar
option = st.sidebar.selectbox("Pilih Analisis", [
    "Heatmap Korelasi",
    "Tren Suhu & Curah Hujan",
    "Klasifikasi Hujan",
    "Clustering Cuaca",
    "Prediksi Suhu Rata-rata"
])

if option == "Heatmap Korelasi":
    st.subheader("Heatmap Korelasi Antar Variabel Iklim")
    st.markdown("""
    Menunjukkan kekuatan hubungan antar variabel numerik menggunakan nilai korelasi Pearson. Nilai mendekati +1 menunjukkan korelasi positif kuat; mendekati -1 berarti negatif kuat.
    """)
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
    st.pyplot(plt)

elif option == "Tren Suhu & Curah Hujan":
    st.subheader("Tren Suhu Rata-rata dan Curah Hujan Harian")
    st.markdown("""
    Menampilkan tren suhu rata-rata harian (garis oranye) dan curah hujan harian (garis biru) dari tahun 2015 hingga 2020 untuk melihat pola musiman dan perubahan cuaca.
    """)
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(df['date'], df['Tavg'], label='Tavg (°C)', color='orange')
    ax.plot(df['date'], df['RR'], label='Curah Hujan (mm)', color='blue')
    ax.legend()
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Nilai")
    st.pyplot(fig)

elif option == "Klasifikasi Hujan":
    st.subheader("Klasifikasi Hari Hujan vs Tidak Hujan")
    st.markdown("""
    Menggunakan model Random Forest untuk memprediksi apakah suatu hari akan hujan atau tidak berdasarkan variabel cuaca seperti suhu, kelembapan, dan durasi sinar matahari.
    """)
    df['rain'] = df['RR'].apply(lambda x: 1 if x > 0 else 0)
    X = df[['Tn', 'Tx', 'RH_avg', 'ss', 'ff_x']]
    y = df['rain']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_rf = RandomForestClassifier(random_state=42)
    model_rf.fit(X_train, y_train)
    acc = model_rf.score(X_test, y_test)
    st.write(f"Akurasi Model: {acc:.2f}")
    st.write("**Pentingnya Fitur**")
    fig, ax = plt.subplots()
    sns.barplot(x=model_rf.feature_importances_, y=X.columns, ax=ax)
    st.pyplot(fig)

elif option == "Clustering Cuaca":
    st.subheader("Clustering Hari Berdasarkan Cuaca")
    st.markdown("""
    Mengelompokkan hari ke dalam 3 cluster berdasarkan kesamaan fitur cuaca menggunakan algoritma KMeans, untuk mengidentifikasi pola seperti hari lembap, panas, atau hujan.
    """)
    X = df[['Tavg', 'RH_avg', 'RR', 'ss', 'ff_avg']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    fig, ax = plt.subplots()
    sns.scatterplot(x='Tavg', y='RH_avg', hue='cluster', data=df, palette='Set1', ax=ax)
    st.pyplot(fig)

elif option == "Prediksi Suhu Rata-rata":
    st.subheader("Prediksi Suhu Rata-rata dengan Linear Regression")
    st.markdown("""
    Menggunakan regresi linier untuk memprediksi suhu rata-rata harian berdasarkan faktor cuaca lain seperti suhu minimum, maksimum, kelembapan, curah hujan, dan angin.
    """)
    X = df[['Tn', 'Tx', 'RH_avg', 'RR', 'ss', 'ff_x', 'ddd_x', 'ff_avg']]
    y = df['Tavg']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.write(f"R2 Score: {r2_score(y_test, y_pred):.2f}")
    st.write(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], '--r')
    ax.set_xlabel("Tavg Aktual")
    ax.set_ylabel("Tavg Prediksi")
    st.pyplot(fig)
