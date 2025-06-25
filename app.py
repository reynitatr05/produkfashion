import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
import datetime
import os

# HARUS PALING ATAS!
st.set_page_config(page_title="Clustering Produk Fashion", layout="wide")

csv_file = "product_fashion.csv"

# Load data dari CSV
def load_data_from_csv():
    base_columns = ["ID", "ProductName", "Price (INR)", "Gender", "PrimaryColor", "ProductBrand", "MinPrice", "MaxPrice", "Timestamp"]
    expected_columns = base_columns + ["Cluster"]
    if not os.path.exists(csv_file) or pd.read_csv(csv_file).empty:
        df_empty = pd.DataFrame(columns=expected_columns)
        df_empty.to_csv(csv_file, index=False)
        return df_empty
    df_loaded = pd.read_csv(csv_file)
    for col in expected_columns:
        if col not in df_loaded.columns:
            df_loaded[col] = None
    df_loaded["Timestamp"] = pd.to_datetime(df_loaded.get("Timestamp", pd.NaT), errors="coerce")
    return df_loaded

# Load awal data utama
if 'df' not in st.session_state:
    st.session_state.df = load_data_from_csv()

# Inisialisasi variabel session state untuk menyimpan produk yang baru ditambahkan untuk tampilan
# Ini akan menyimpan produk yang ditambahkan dalam satu sesi Streamlit
if 'newly_added_products_display' not in st.session_state:
    st.session_state.newly_added_products_display = pd.DataFrame(columns=["ID", "ProductName", "Price (INR)", "Gender", "PrimaryColor", "ProductBrand", "MinPrice", "MaxPrice", "Timestamp", "Cluster"])

df = st.session_state.df

# Sidebar navigasi
st.sidebar.title("Klasterisasi Produk Fashion")
page = st.sidebar.radio("Pilih Halaman", ["📊 Dashboard", "🧠 Clustering Produk", "➕ Tambah Produk"])

# ---
# 📊 DASHBOARD
# ---
if page == "📊 Dashboard":
    st.title("📊 Dashboard Produk Fashion")

    gender_filter = st.selectbox("Filter Gender", ["All", "Men", "Women"])
    if gender_filter == "All":
        df_dash = df.copy()
    else:
        df_dash = df[df["Gender"] == gender_filter]

    if df_dash.empty:
        st.warning("⚠️ Belum ada data yang bisa ditampilkan.")
    else:
        with st.expander("📄 Lihat Semua Data"):
            st.dataframe(df_dash)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Distribusi Gender")
            fig1, ax1 = plt.subplots()
            sns.countplot(data=df_dash, x="Gender", palette="Set2", ax=ax1)
            st.pyplot(fig1)

        with col2:
            st.subheader("Distribusi Harga")
            fig2, ax2 = plt.subplots()
            sns.histplot(df_dash["Price (INR)"], bins=30, kde=True, ax=ax2, color='skyblue')
            st.pyplot(fig2)

# ---
# 🧠 CLUSTERING
# ---
elif page == "🧠 Clustering Produk":
    st.title("🧠 Clustering Produk berdasarkan Harga dan Gender")

    df_cluster = df[df["Gender"].isin(["Men", "Women"])].copy()
    if df_cluster.empty:
        st.warning("⚠️ Tidak ada data 'Men' atau 'Women' yang bisa dikelompokkan. Harap tambahkan data terlebih dahulu.")
        st.stop()

    le = LabelEncoder()
    # Handle unknown classes during transform if not all genders are present in training
    try:
        df_cluster["Gender_encoded"] = le.fit_transform(df_cluster["Gender"])
    except ValueError as e:
        st.error(f"Error encoding gender: {e}. Pastikan ada setidaknya dua gender 'Men'/'Women' yang berbeda.")
        st.stop()

    X = df_cluster[["Gender_encoded", "Price (INR)"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Pastikan jumlah cluster tidak melebihi jumlah sampel yang tersedia
    max_clusters = min(6, len(df_cluster))
    if max_clusters < 2:
        st.warning("⚠️ Tidak cukup data untuk melakukan clustering (minimal 2 sampel diperlukan).")
        st.stop()

    n_clusters = st.slider("Jumlah Cluster", 2, max_clusters, 3)
    
    # Menambahkan penanganan error untuk KMeans jika n_clusters terlalu besar untuk data
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df_cluster["Cluster"] = kmeans.fit_predict(X_scaled)
    except Exception as e:
        st.error(f"⚠️ Terjadi kesalahan saat melakukan clustering. Coba kurangi jumlah cluster atau tambahkan lebih banyak data. Error: {e}")
        st.stop()


    # Simpan model ke session_state
    st.session_state["label_encoder"] = le
    st.session_state["standard_scaler"] = scaler
    st.session_state["kmeans_model"] = kmeans

    # Update hasil cluster ke dataframe utama berdasarkan ID
    # Menggunakan .loc untuk menghindari SettingWithCopyWarning dan memastikan update yang benar
    for i, row in df_cluster.iterrows():
        # Pastikan kolom 'ID' ada dan cocok untuk merging
        if 'ID' in st.session_state.df.columns and 'ID' in row:
            st.session_state.df.loc[st.session_state.df["ID"] == row["ID"], "Cluster"] = row["Cluster"]
    
    st.session_state.df.to_csv(csv_file, index=False)

    st.success("✅ Clustering selesai.")
    st.dataframe(df_cluster[["ProductName", "Price (INR)", "Gender", "Cluster"]])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df_cluster, x="Price (INR)", y="Gender_encoded", hue="Cluster", palette="viridis", s=80, ax=ax)
    ax.set_ylabel("Gender (encoded)")
    # Mengubah label y-axis menjadi label gender asli
    gender_ticks = [0, 1] # Asumsi 0 dan 1 untuk "Men" dan "Women" setelah encoding
    gender_labels = le.inverse_transform(gender_ticks)
    ax.set_yticks(gender_ticks)
    ax.set_yticklabels(gender_labels)
    st.pyplot(fig)

# ---
# ➕ TAMBAH PRODUK
# ---
elif page == "➕ Tambah Produk":
    st.title("➕ Tambah Produk Baru")

    if st.button("🚨 Reset Data CSV"):
        if os.path.exists(csv_file):
            os.remove(csv_file)
            st.session_state.df = load_data_from_csv()
            # Juga reset tampilan produk yang baru ditambahkan
            st.session_state.newly_added_products_display = pd.DataFrame(columns=["ID", "ProductName", "Price (INR)", "Gender", "PrimaryColor", "ProductBrand", "MinPrice", "MaxPrice", "Timestamp", "Cluster"])
            st.success("✅ File berhasil direset.")
            st.rerun() # Memuat ulang aplikasi untuk mencerminkan perubahan

    with st.form("form_input"):
        nama_produk = st.text_input("Nama Produk")
        harga_input = st.text_input("Harga Produk (INR)", "500")
        harga_min_input = st.text_input("Harga Minimum", "0")
        harga_max_input = st.text_input("Harga Maksimum", "0")
        gender = st.selectbox("Gender", ["Men", "Women")
        warna = st.text_input("Warna Produk (opsional)")
        brand = st.text_input("Brand (opsional)")
        submit = st.form_submit_button("Tambah Produk")

        # Validasi input numerik
        try:
            harga = int(harga_input)
            harga_min = int(harga_min_input)
            harga_max = int(harga_max_input)
        except ValueError:
            st.error("⚠️ Masukkan angka yang valid untuk harga.")
            st.stop() # Hentikan eksekusi jika input tidak valid

        if submit:
            if nama_produk.strip() == "":
                st.error("❌ Nama produk tidak boleh kosong.")
            else:
                # Otomatis latih model jika belum tersedia atau jika datanya berubah signifikan
                # Cek apakah model sudah dilatih dan masih valid
                model_ready = "kmeans_model" in st.session_state and \
                              "label_encoder" in st.session_state and \
                              "standard_scaler" in st.session_state

                if not model_ready:
                    df_train = st.session_state.df[st.session_state.df["Gender"].isin(["Men", "Women"])].copy()
                    if len(df_train) >= 2: # Minimal 2 sampel untuk clustering
                        le = LabelEncoder()
                        df_train["Gender_encoded"] = le.fit_transform(df_train["Gender"])
                        X = df_train[["Gender_encoded", "Price (INR)"]]
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        kmeans = KMeans(n_clusters=min(3, len(df_train)), random_state=42, n_init=10) # Sesuaikan n_clusters
                        kmeans.fit(X_scaled)
                        st.session_state["label_encoder"] = le
                        st.session_state["standard_scaler"] = scaler
                        st.session_state["kmeans_model"] = kmeans
                        st.info("💡 Model clustering telah dilatih secara otomatis.")
                        model_ready = True
                    else:
                        st.warning("⚠️ Tidak cukup data 'Men' atau 'Women' untuk melatih model clustering secara otomatis. Harap tambahkan minimal 2 produk 'Men' atau 'Women' dan jalankan clustering secara manual di halaman 'Clustering Produk' terlebih dahulu.")
                
                cluster_pred = None
                if gender in ["Men", "Women"] and model_ready:
                    try:
                        le = st.session_state["label_encoder"]
                        scaler = st.session_state["standard_scaler"]
                        kmeans = st.session_state["kmeans_model"]
                        
                        # Pastikan gender yang diinput ada di kelas yang dikenal LabelEncoder
                        if gender in le.classes_:
                            gender_encoded = le.transform([gender])[0]
                            data_to_predict = pd.DataFrame([[gender_encoded, harga]], columns=["Gender_encoded", "Price (INR)"])
                            data_scaled = scaler.transform(data_to_predict)
                            cluster_pred = int(kmeans.predict(data_scaled)[0])
                            st.info(f"Prediksi Cluster: {cluster_pred}")
                        else:
                            st.warning(f"⚠️ Gender '{gender}' belum ada di data yang digunakan untuk melatih model. Tidak dapat memprediksi cluster untuk gender ini.")
                    except Exception as e:
                        st.warning(f"⚠️ Gagal memprediksi cluster untuk produk ini. Pastikan model sudah dilatih dengan data yang relevan. Error: {e}")
                elif gender not in ["Men", "Women"]:
                    st.info("ℹ️ Prediksi cluster hanya relevan untuk produk dengan gender 'Men' atau 'Women'.")
                elif not model_ready:
                    st.info("ℹ️ Model clustering belum siap. Harap latih model di halaman 'Clustering Produk' terlebih dahulu atau tambahkan lebih banyak produk 'Men' atau 'Women'.")


                data_baru = {
                    "ID": str(datetime.datetime.now().timestamp()), # ID unik berdasarkan timestamp
                    "ProductName": nama_produk,
                    "Price (INR)": harga,
                    "Gender": gender,
                    "PrimaryColor": warna if warna else "Unknown",
                    "ProductBrand": brand if brand else "Unknown",
                    "MinPrice": harga_min,
                    "MaxPrice": harga_max,
                    "Timestamp": datetime.datetime.now(),
                    "Cluster": cluster_pred
                }

                # Tambahkan produk baru ke dataframe utama
                st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame([data_baru])], ignore_index=True)
                st.session_state.df.to_csv(csv_file, index=False)
                
                # Tambahkan produk baru ke dataframe untuk tampilan "Produk yang Baru Ditambahkan"
                st.session_state.newly_added_products_display = pd.concat([st.session_state.newly_added_products_display, pd.DataFrame([data_baru])], ignore_index=True)

                st.success(f"✅ Produk '{nama_produk}' berhasil ditambahkan.")
                
    st.subheader("Produk yang Baru Ditambahkan")
    if not st.session_state.newly_added_products_display.empty:
        # Tampilkan produk yang baru ditambahkan dalam sesi ini
        st.dataframe(st.session_state.newly_added_products_display)
    else:
        st.info("Belum ada produk yang ditambahkan dalam sesi ini.")
