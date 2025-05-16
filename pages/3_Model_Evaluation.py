import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Setting layout halaman
st.set_page_config(
        page_title="SLA-Indonesia",
        page_icon="🏠",
        layout="centered",
        initial_sidebar_state="expanded"
    )

# Load custom CSS file
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# judul section
st.title('GCM Evaluation')

# st.divider()

# --- Fungsi untuk memuat data evaluasi dengan cache ---
@st.cache_data
def load_evaluation_data(sheet_name):
    return pd.read_excel("evaluasi_model_lengkap_ok.xlsx", sheet_name=sheet_name)

@st.cache_data
def load_all_evaluation_data(sheet_map):
    data_list = []
    for model_name, sheet_name in sheet_map.items():
        df = pd.read_excel("evaluasi_model_lengkap_ok.xlsx", sheet_name=sheet_name)
        df["Model"] = model_name
        data_list.append(df)
    return pd.concat(data_list, ignore_index=True)

# --- UI untuk Pilihan Model dan GCM ---
tab1, tab2, tab3 = st.tabs([
    "📈 Evaluation Per Metric",
    "📉 Evaluation Per GCM",
    "📊 Table of Evaluation"
])

with tab1:
    model_type = st.selectbox("Pilihan Model", ["Original", "CNN", "CNN-LSTM"])
    selected_gcm = st.selectbox("Pilihan GCM", ["CanESM5", "EC-Earth", "MIROC6", "MPI", "ACCESS", "ACCESS-CM2", "CMCC"])

    # --- Mapping nama sheet dan nama GCM ---
    sheet_map = {
        "Original": "Sheet1",
        "CNN": "Sheet2",
        "CNN-LSTM": "Sheet3"
    }

    gcm_map = {
        'CanESM5': 'canesm5',
        'EC-Earth': 'ec-earth',
        'MIROC6': 'miroc6',
        'MPI': 'mpi',
        'ACCESS': 'access',
        'ACCESS-CM2': 'access-cm2',
        'CMCC': 'cmcc'
    }

    # --- Load Sheet Sesuai Pilihan ---
    sheet_name = sheet_map[model_type]
    eval_excel = load_evaluation_data(sheet_name)

    # --- Ambil Baris Sesuai GCM ---
    selected_gcm_lower = gcm_map[selected_gcm]
    gcm_row = eval_excel[eval_excel['GCM'] == selected_gcm_lower].iloc[0]

    # --- Tampilkan Metrik Evaluasi ---
    st.subheader(f"📈 Metric Evaluasi dari {selected_gcm} model {model_type}")

    cor = gcm_row['Correlation']
    rmse = gcm_row['RMSE']
    bias = gcm_row['Bias']  # Atau bisa diganti jadi MAE kalau kamu prefer yang absolut

    col1, col2, col3 = st.columns(3)
    col1.metric("Correlation", f"{cor:.3f}")
    col2.metric("RMSE", f"{rmse:.3f} m")
    col3.metric("Bias", f"{bias:.3f} m")  # Ganti label & variabel jika ingin pakai MAE

    if cor > 0.8:
        st.success("👍 Very Good Correlation!")
    elif cor > 0.6:
        st.info("👌 Fairly Correlation")
    else:
        st.warning("⚠️ Poor Correlation")

    # --- Grafik Evaluasi GCM ---
    st.subheader(f"📊 Grafik Evaluasi dari GCM versi {model_type}")

    df_ranked = eval_excel.copy()
    df_ranked = df_ranked.sort_values(by="Correlation", ascending=False).reset_index(drop=True)
    df_ranked.index += 1  # Mulai dari 1

    metric_option = st.selectbox("Pilihan Metric:", ["Correlation", "RMSE", "Bias"])

    fig2 = px.bar(
        df_ranked,
        x="GCM",
        y=metric_option,
        color="GCM",
        text=df_ranked[metric_option].round(2),
        color_discrete_sequence=px.colors.qualitative.Pastel
    )

    fig2.update_layout(
        xaxis_title="GCM",
        yaxis_title=metric_option,
        title={'text':f"Nila {metric_option} untuk semua GCM pada model {model_type} model version",
               'x': 0.5, 'y': 0.9, 'xanchor': 'center', 'yanchor': 'top',
                'font': {'size': 20, 'family': 'Arial, sans-serif'}},
        showlegend=False
    )
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.subheader("📊 Radar Chart untuk Perbandingan Model")
    # --- Load semua sheet ---
    sheet_map = {
        "Original": "Sheet1",
        "CNN": "Sheet2",
        "CNN-LSTM": "Sheet3"
    }
    model_list = list(sheet_map.keys())

    # Gabungkan semua sheet menjadi satu DataFrame
    df_all = load_all_evaluation_data(sheet_map)

    # --- Pilih GCM ---
    available_gcms = df_all["GCM"].str.upper().unique().tolist()
    model_to_plot = st.multiselect("Daftar Model:", model_list, default=model_list)
    gcm_to_plot = st.selectbox("Pilihan GCM:", available_gcms)

    radar_data = df_all[df_all["Model"].isin(model_to_plot) & (df_all["GCM"].str.upper() == gcm_to_plot)]

    categories = ["Correlation", "RMSE", "Bias"]

    # Warna tetap per model
    color_map = {
        "Original": "#1f77b4",  # biru
        "CNN": "#2ca02c",       # hijau
        "CNN-LSTM": "#d62728"   # merah
    }

    fig_radar = go.Figure()

    for _, row in radar_data.iterrows():
        model = row["Model"]
        values = [row[cat] for cat in categories]
        fig_radar.add_trace(go.Scatterpolar(
            r=[row[cat] for cat in categories],
            theta=categories,
            fill='toself',
            name=model,
            line=dict(color=color_map.get(model, "#888")),
            marker=dict(size=6),
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True,
        width=600,
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=12)
        ),
        title={'text':f"Metric Evaluasi dari {gcm_to_plot}",
               'x': 0.5, 'y': 0.9, 'xanchor': 'center', 'yanchor': 'top',
                'font': {'size': 20, 'family': 'Arial, sans-serif'}},
    )
    st.plotly_chart(fig_radar, use_container_width=True)

with tab3:
    # --- Load semua sheet ---
    sheet_map = {
        "Original": "Sheet1",
        "CNN": "Sheet2",
        "CNN-LSTM": "Sheet3"
    }
    model_list = list(sheet_map.keys())
    data_list = []

    for model_name, sheet_name in sheet_map.items():
        df = load_evaluation_data(sheet_name)
        df["Model"] = model_name
        data_list.append(df)

    # Gabungkan semua sheet menjadi satu DataFrame
    df_all = pd.concat(data_list, ignore_index=True)

    # Normalisasi nama kolom
    df_all.columns = df_all.columns.str.strip()
    
    # Konversi ke numerik
    df_all["Correlation"] = pd.to_numeric(df_all["Correlation"], errors='coerce')
    df_all["RMSE"] = pd.to_numeric(df_all["RMSE"], errors='coerce')
    
    # Drop baris yang NaN jika perlu
    df_all = df_all.dropna(subset=["Correlation", "RMSE"])

    st.subheader("📋 Table of Evaluation")
    st.dataframe(
        df_all.style.background_gradient(cmap='YlGn', subset=["Correlation"])
                    .background_gradient(cmap='OrRd_r', subset=["RMSE"]),
    use_container_width=True, height=780)

# membuat narasi tabel dalam keterangan
with st.expander(":blue-background[Description :]"):
    st.caption(('''**Metric Evaluasi** menggunakan "Correlation", "RMSE", "Bias", dan "MAE".'''))
    st.caption(('''Membandingkan evaluasi antara GCM dengan semua metrik yang tersedia.'''))
    st.caption(('''Membandingkan model yang lebih efektif dalam menangani data grid spasial, khususnya variabel SLA.'''))

