import streamlit as st
import pandas as pd
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
import os
import numpy as np
from datetime import date, datetime

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("Aplikasi Perhitungan SHAP Rata-rata")

st.markdown(
    """
    Aplikasi ini menghitung SHAP values rata-rata dari data ERA5 dan Curah Hujan 
    yang Anda unggah. Hasil plot SHAP rata-rata akan ditampilkan langsung, 
    dan SHAP values dalam bentuk tabel akan disimpan ke file Excel.
    """
)

# --- Upload Bagian File ---
st.sidebar.header("Unggah File Data")

uploaded_era5_file = st.sidebar.file_uploader("Unggah File ERA5 (Excel, .xlsx)", type=["xlsx"])
uploaded_ch_file = st.sidebar.file_uploader("Unggah File Curah Hujan (CSV, .csv)", type=["csv"])

df_era5_preview = None
df_ch_preview = None

if uploaded_era5_file is not None:
    try:
        df_era5_preview = pd.read_excel(uploaded_era5_file)
        st.sidebar.success("File ERA5 berhasil diunggah!")
        st.sidebar.dataframe(df_era5_preview.head())
    except Exception as e:
        st.sidebar.error(f"Error membaca file ERA5: {e}")

if uploaded_ch_file is not None:
    try:
        df_ch_preview = pd.read_csv(uploaded_ch_file)
        st.sidebar.success("File Curah Hujan berhasil diunggah!")
        st.sidebar.dataframe(df_ch_preview.head())
    except Exception as e:
        st.sidebar.error(f"Error membaca file Curah Hujan: {e}")

# --- Bagian Deteksi Variabel ---
st.sidebar.header("Deteksi Variabel")
selected_features = []

if df_era5_preview is not None and df_ch_preview is not None:
    required_era5_cols = ['lat', 'lon'] # Asumsi variabel fitur lainnya adalah yang tersisa
    required_ch_cols = ['lat', 'lon', 'rainfall']

    # Cek kolom yang diperlukan
    era5_cols_missing = [col for col in required_era5_cols if col not in df_era5_preview.columns]
    ch_cols_missing = [col for col in required_ch_cols if col not in df_ch_preview.columns]

    if era5_cols_missing:
        st.sidebar.error(f"File ERA5 kehilangan kolom wajib: {', '.join(era5_cols_missing)}")
    if ch_cols_missing:
        st.sidebar.error(f"File Curah Hujan kehilangan kolom wajib: {', '.join(ch_cols_missing)}")

    if not era5_cols_missing and not ch_cols_missing:
        st.sidebar.subheader("Pilih Fitur")
        all_era5_cols = df_era5_preview.drop(columns=['lat', 'lon'], errors='ignore').columns.tolist()
        selected_features = st.sidebar.multiselect(
            "Pilih Variabel Fitur (dari ERA5):",
            options=all_era5_cols,
            default=all_era5_cols # Pilih semua secara default
        )
        if not selected_features:
            st.sidebar.warning("Harap pilih setidaknya satu variabel fitur dari ERA5.")
    else:
        st.sidebar.warning("Harap unggah kedua file dan pastikan memiliki kolom yang diperlukan.")
else:
    st.sidebar.warning("Harap unggah kedua file untuk deteksi variabel.")

st.sidebar.markdown("---")

# Tombol untuk memulai perhitungan
if st.sidebar.button("Mulai Perhitungan SHAP Rata-rata"):
    if uploaded_era5_file is None or uploaded_ch_file is None:
        st.error("Harap unggah kedua file (ERA5 dan Curah Hujan) sebelum memulai perhitungan.")
    elif not selected_features:
        st.error("Harap pilih setidaknya satu variabel fitur sebelum memulai perhitungan.")
    else:
        st.info("Memulai perhitungan SHAP. Proses ini mungkin memakan waktu...")

        # --- Fungsi calculate_and_average_shap (dimodifikasi) ---
        @st.cache_data # Cache hasil untuk performa yang lebih baik jika input tidak berubah
        def calculate_average_shap_for_uploaded_files(df_era5_data, df_ch_data, features_list):
            st.subheader("Detail Proses Data Gabungan")
            
            df_var = df_era5_data.copy()
            df_ch = df_ch_data.copy()

            df_var['lat'] = df_var['lat'].round(3)
            df_var['lon'] = df_var['lon'].round(3)
            df_ch['lat'] = df_ch['lat'].round(3)
            df_ch['lon'] = df_ch['lon'].round(3)

            koordinat_ch = df_ch[['lat', 'lon']].drop_duplicates()
            df_var_filtered = pd.merge(df_var, koordinat_ch, on=['lat', 'lon'], how='inner')

            df_merge = pd.merge(df_var_filtered, df_ch[['lat', 'lon', 'rainfall']], on=['lat', 'lon'], how='inner')

            if df_merge.empty:
                st.error("Data gabungan kosong. Tidak dapat melakukan perhitungan SHAP.")
                return None, None

            # Pastikan hanya fitur yang dipilih yang digunakan
            X = df_merge[features_list]
            y = df_merge['rainfall']

            if len(X) == 0:
                st.error("Tidak ada fitur yang valid setelah penggabungan. Tidak dapat melakukan perhitungan SHAP.")
                return None, None
            if len(y) == 0:
                st.error("Tidak ada target (rainfall) yang valid setelah penggabungan. Tidak dapat melakukan perhitungan SHAP.")
                return None, None

            st.write(f"Jumlah baris data yang akan diproses: {len(X)}")
            
            model = xgb.XGBRegressor(n_estimators=100, max_depth=4, random_state=42)
            model.fit(X, y)

            explainer = shap.Explainer(model)
            shap_values_obj = explainer(X)

            # Karena kita hanya memproses satu set data (dari file yang diunggah),
            # SHAP values ini sudah bisa dianggap sebagai "rata-rata" untuk dataset tersebut.
            # Jika ada banyak data poin, SHAP summary plot akan menunjukkan rata-ratanya.
            
            averaged_shap_values = shap_values_obj.values
            final_X_features = X.columns.tolist()

            # Buat Explanation object untuk plot
            averaged_shap_explanation = shap.Explanation(
                values=averaged_shap_values,
                base_values=explainer.expected_value,
                data=X.values,
                feature_names=final_X_features
            )

            # Buat plot SHAP Summary
            fig_avg, ax_avg = plt.subplots(figsize=(12, 8))
            shap.summary_plot(averaged_shap_explanation, X, show=False, ax=ax_avg)
            plt.title(f"SHAP Summary Plot Rata-rata")
            plt.tight_layout()
            
            return fig_avg, averaged_shap_explanation.values, final_X_features

        # Panggil fungsi utama dengan DataFrame yang diunggah
        fig_result, shap_values_array, feature_names = calculate_average_shap_for_uploaded_files(
            df_era5_preview, df_ch_preview, selected_features
        )

        if fig_result is not None:
            st.success("✅ Perhitungan SHAP selesai!")
            st.subheader("Plot SHAP Rata-rata Keseluruhan")
            st.pyplot(fig_result)
            plt.close(fig_result) # Tutup figure untuk membebaskan memori

            if shap_values_array is not None and feature_names is not None:
                st.subheader("SHAP Values Rata-rata (Excel)")
                # Hitung rata-rata absolut SHAP value untuk setiap fitur
                mean_abs_shap_values = np.mean(np.abs(shap_values_array), axis=0)
                
                # Buat DataFrame dari rata-rata absolut SHAP values
                df_shap_results = pd.DataFrame({
                    'Feature': feature_names,
                    'Mean_Absolute_SHAP_Value': mean_abs_shap_values
                }).sort_values(by='Mean_Absolute_SHAP_Value', ascending=False).reset_index(drop=True)

                st.dataframe(df_shap_results)

                # Simpan ke Excel dan sediakan link download
                excel_buffer = pd.io.common.BytesIO()
                df_shap_results.to_excel(excel_buffer, index=False, sheet_name='SHAP_Values_Rata_rata')
                excel_buffer.seek(0)

                st.download_button(
                    label="📥 Unduh SHAP Values Rata-rata sebagai Excel",
                    data=excel_buffer,
                    file_name="shap_values_rata_rata.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            st.error("Perhitungan SHAP tidak berhasil. Periksa pesan kesalahan di atas.")

st.markdown("---")
st.markdown("Aplikasi ini memproses file yang diunggah sebagai satu kesatuan data untuk menghitung SHAP values rata-rata.")