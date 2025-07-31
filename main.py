import streamlit as st
import pandas as pd
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
import os
import numpy as np
import io # Import io for BytesIO

# --- Streamlit UI Configuration ---
st.set_page_config(
    page_title="Aplikasi Perhitungan SHAP",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Aplikasi Perhitungan SHAP Rata-rata")

st.markdown(
    """
    Unggah satu file (Excel atau CSV) yang berisi data Anda. Aplikasi ini akan memandu Anda untuk memilih 
    **variabel target** dan **variabel fitur** untuk analisis SHAP. 
    Hasil plot SHAP rata-rata akan ditampilkan langsung, dan SHAP values dalam bentuk tabel 
    akan dapat diunduh sebagai file Excel.
    """
)

# --- File Upload Section ---
st.sidebar.header("📤 Unggah File Data")

uploaded_file = st.sidebar.file_uploader(
    "Pilih file data Anda (Excel (.xlsx) atau CSV (.csv)):", 
    type=["xlsx", "csv"],
    help="Pastikan file Anda bersih dan siap dianalisis."
)

df_data = None # Initialize df_data here

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.xlsx'):
            df_data = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.csv'):
            df_data = pd.read_csv(uploaded_file)
        
        st.sidebar.success("✅ File berhasil diunggah!")
        st.sidebar.subheader("Pratinjau Data (5 Baris Pertama):")
        st.sidebar.dataframe(df_data.head())
    except Exception as e:
        st.sidebar.error(f"❌ Error membaca file: Pastikan format file benar dan tidak rusak. Detail: `{e}`")

# --- Variable Selection Section ---
st.sidebar.header("⚙️ Pilih Variabel")

target_variable = None
feature_variables = []

if df_data is not None:
    available_columns = df_data.columns.tolist()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Pilih Variabel Target (Y)")
    target_variable = st.sidebar.selectbox(
        "Pilih kolom yang ingin Anda prediksi (variabel target):",
        options=["-- Pilih --"] + available_columns, # Add a default empty option
        index=available_columns.index('rainfall') + 1 if 'rainfall' in available_columns else 0, # Default 'rainfall' or first empty option
        help="Ini adalah variabel dependen yang ingin Anda jelaskan."
    )
    
    # Remove the placeholder if it was selected
    if target_variable == "-- Pilih --":
        target_variable = None

    # Filter available columns for features based on selected target
    potential_features = [col for col in available_columns if col != target_variable]

    st.sidebar.subheader("Pilih Variabel Fitur (X)")
    
    # Try to default to 'lat' and 'lon' if they exist, along with all other potential features
    default_feature_selection = []
    if 'lat' in potential_features: default_feature_selection.append('lat')
    if 'lon' in potential_features: default_feature_selection.append('lon')
    
    # Add all other potential features to the default selection
    remaining_features = [col for col in potential_features if col not in ['lat', 'lon']]
    default_feature_selection.extend(remaining_features)

    feature_variables = st.sidebar.multiselect(
        "Pilih kolom yang akan digunakan sebagai input (variabel fitur):",
        options=potential_features,
        default=default_feature_selection,
        help="Ini adalah variabel independen yang akan digunakan untuk menjelaskan variabel target."
    )

    if not feature_variables and target_variable: # Only warn if target is selected but no features
        st.sidebar.warning("⚠️ Harap pilih setidaknya satu variabel fitur.")
    if not target_variable and df_data is not None: # Only warn if file uploaded but target not selected
        st.sidebar.warning("⚠️ Harap pilih variabel target.")

st.sidebar.markdown("---")

# --- Run Calculation Button ---
if st.sidebar.button("🚀 Mulai Perhitungan SHAP Rata-rata"):
    if uploaded_file is None:
        st.error("❌ Harap unggah file data Anda terlebih dahulu.")
    elif target_variable is None:
        st.error("❌ Harap pilih variabel target Anda.")
    elif not feature_variables:
        st.error("❌ Harap pilih setidaknya satu variabel fitur.")
    else:
        with st.spinner("⏳ Memulai perhitungan SHAP... Proses ini mungkin memakan waktu tergantung ukuran data Anda."):
            # --- Function to Calculate Average SHAP ---
            # Removed @st.cache_data here temporarily to rule out caching issues if any
            def calculate_average_shap_for_single_file(df_input_data, target_col, feature_cols):
                st.subheader("Detail Proses Data")
                
                df_processed = df_input_data.copy()

                missing_features = [col for col in feature_cols if col not in df_processed.columns]
                if missing_features:
                    st.error(f"❌ Kolom fitur berikut tidak ditemukan dalam data: {', '.join(missing_features)}. Harap periksa pilihan Anda.")
                    return None, None, None, None 

                if target_col not in df_processed.columns:
                    st.error(f"❌ Kolom target '{target_col}' tidak ditemukan dalam data. Harap periksa pilihan Anda.")
                    return None, None, None, None 

                if 'lat' in feature_cols and 'lat' in df_processed.columns:
                    df_processed['lat'] = df_processed['lat'].round(3)
                if 'lon' in feature_cols and 'lon' in df_processed.columns:
                    df_processed['lon'] = df_processed['lon'].round(3)

                for col in feature_cols:
                    df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                
                df_processed[target_col] = pd.to_numeric(df_processed[target_col], errors='coerce')

                X = df_processed[feature_cols]
                y = df_processed[target_col]

                initial_rows = len(X)
                
                # Combine X and y, then drop NaNs
                combined_df_for_dropna = pd.concat([X, y], axis=1)
                combined_df_cleaned = combined_df_for_dropna.dropna()
                
                X_cleaned = combined_df_cleaned[feature_cols]
                y_cleaned = combined_df_cleaned[target_col]

                if len(X_cleaned) == 0:
                    st.error("❌ Tidak ada data yang valid setelah membersihkan nilai yang hilang (NaN) atau non-numerik. Perhitungan SHAP tidak dapat dilakukan. Pastikan kolom fitur dan target hanya berisi nilai numerik.")
                    return None, None, None, None
                if len(y_cleaned) == 0:
                    st.error("❌ Variabel target tidak memiliki nilai yang valid setelah membersihkan nilai yang hilang (NaN) atau non-numerik. Perhitungan SHAP tidak dapat dilakukan.")
                    return None, None, None, None
                
                st.info(f"Jumlah baris data yang akan diproses setelah membersihkan nilai hilang atau non-numerik: **{len(X_cleaned)}** dari **{initial_rows}** baris awal.")
                
                # --- Train Model ---
                model = None # Initialize model
                try:
                    model = xgb.XGBRegressor(n_estimators=100, max_depth=4, random_state=42)
                    model.fit(X_cleaned, y_cleaned) # Use cleaned data for training
                    st.success("✅ Model XGBoost berhasil dilatih.")
                except Exception as e:
                    st.error(f"❌ Gagal melatih model XGBoost. Mungkin ada masalah dengan data atau pilihan variabel Anda. Detail: `{e}`")
                    return None, None, None, None

                # --- Calculate SHAP Values ---
                shap_values_array = None
                expected_value = None
                final_X_features = X_cleaned.columns.tolist() # Get feature names from cleaned X
                
                try:
                    # Use a TreeExplainer for XGBoost for better performance and accuracy
                    explainer = shap.TreeExplainer(model) 
                    shap_values_obj = explainer(X_cleaned) # This computes the Explanation object
                    
                    shap_values_array = shap_values_obj.values # Extract numpy array of SHAP values
                    expected_value = explainer.expected_value # Extract base value

                    st.success("✅ SHAP values berhasil dihitung.")
                except Exception as e:
                    st.error(f"❌ Gagal menghitung SHAP values. Pastikan data tidak kosong atau memiliki masalah numerik setelah pembersihan. Detail: `{e}`")
                    return None, None, None, None
                
                # Manually construct shap.Explanation to ensure consistency for plotting
                # This explicitly sets data and feature_names which sometimes helps resolve TypeErrors
                shap_explanation_for_plot = shap.Explanation(
                    values=shap_values_array,
                    base_values=expected_value,
                    data=X_cleaned.values, # Ensure this is a numpy array
                    feature_names=final_X_features
                )
                
                return shap_values_array, final_X_features, shap_explanation_for_plot

            # Call the main SHAP calculation function
            # We now expect 3 return values: raw SHAP values, feature names, and the Explanation object
            raw_shap_values, feature_names_for_table, shap_explanation_object = calculate_average_shap_for_single_file(
                df_data, target_variable, feature_variables
            )

            # Check if SHAP calculation was successful (by checking the explanation object)
            if shap_explanation_object is not None:
                st.success("✅ Analisis SHAP Selesai!")
                st.subheader(f"📈 Plot SHAP Rata-rata Keseluruhan untuk Target: **{target_variable}**")
                
                # --- Generate and display SHAP Summary Plot ---
                fig_avg, ax_avg = plt.subplots(figsize=(12, 8))
                
                # Directly use the Explanation object obtained from the function
                shap.summary_plot(shap_explanation_object, show=False, ax=ax_avg) 
                
                plt.title(f"SHAP Summary Plot Rata-rata untuk Target: {target_variable}")
                plt.tight_layout()
                st.pyplot(fig_avg)
                plt.close(fig_avg) # Close the figure to free up memory

                if raw_shap_values is not None and feature_names_for_table is not None:
                    st.subheader("📊 SHAP Values Rata-rata (Tabel & Unduh Excel)")
                    
                    # Calculate mean absolute SHAP value for each feature
                    mean_abs_shap_values = np.mean(np.abs(raw_shap_values), axis=0)
                    
                    # Create DataFrame from mean absolute SHAP values
                    df_shap_results = pd.DataFrame({
                        'Feature': feature_names_for_table,
                        'Mean_Absolute_SHAP_Value': mean_abs_shap_values
                    }).sort_values(by='Mean_Absolute_SHAP_Value', ascending=False).reset_index(drop=True)

                    st.dataframe(df_shap_results)

                    # Provide download button for Excel
                    excel_buffer = io.BytesIO()
                    df_shap_results.to_excel(excel_buffer, index=False, sheet_name=f'SHAP_Values_{target_variable}')
                    excel_buffer.seek(0)

                    st.download_button(
                        label="📥 Unduh SHAP Values Rata-rata sebagai Excel",
                        data=excel_buffer,
                        file_name=f"shap_values_rata_rata_{target_variable}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            else:
                st.error("❌ Perhitungan SHAP tidak berhasil diselesaikan. Harap periksa pesan kesalahan di atas untuk detail.")