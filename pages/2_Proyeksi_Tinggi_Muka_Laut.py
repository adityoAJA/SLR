import streamlit as st
import os
import re
import pandas as pd
import xarray as xr
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import linregress
import requests
import io
import os
import zipfile

# Setting layout halaman
st.set_page_config(
        page_title="Analisis TML Indonesia",
        page_icon="🏠",
        layout="centered",
        initial_sidebar_state="expanded"
    )

# Load custom CSS file
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Fungsi untuk mengunduh dan mengekstrak file ZIP dari GitHub Release
@st.cache_resource
def download_and_extract_zip(url, output_folder):
    if not os.path.exists(output_folder) or not os.listdir(output_folder):
        os.makedirs(output_folder, exist_ok=True)
        try:
            response = requests.get(url)
            response.raise_for_status()  # Memastikan respons OK
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                zip_ref.extractall(output_folder)
        except requests.exceptions.RequestException as e:
            st.error(f"Gagal mengunduh file: {e}")
        except zipfile.BadZipFile:
            st.error("File ZIP rusak atau tidak valid")
    return output_folder

# URL dan folder target untuk data trend dan hasil
url1 = "https://github.com/adityoAJA/SLR/releases/download/v1/data_trend.zip"
output_folder1 = "trend"
url2 = "https://github.com/adityoAJA/SLR/releases/download/v1/data_tahunan.zip"
output_folder2 = "hasil"

# judul section
st.title('Proyeksi Tinggi Muka Laut')

# st.divider()

tab1, tab2 = st.tabs(["📈 Rata-Rata Tinggi Muka Laut Per Tahun", "📊 Tren Tinggi Muka Laut"])

# ====================
# TAB 1: RATA-RATA TAHUNAN
# ====================
with tab1:
    # st.subheader("Peta Rata-Rata Tahunan Tinggi Muka Laut")
    
    # Jalankan fungsi
    with st.spinner("Sedang mengunduh dan mengekstrak data rata-rata tahunan..."):
        folder_path2 = download_and_extract_zip(url2, output_folder2)

    # Folder hasil prediksi
    folder_path = "hasil/"

    # Ambil semua nama file di folder
    files = [f for f in os.listdir(folder_path) if f.endswith(".nc")]

    # Ekstrak komponen: gcm, skenario, metode
    gcm_list = []
    skenario_list = []
    metode_list = []

    pattern = r"(.+?)_(ssp\d{3})_2021_2100_(.+?)\.nc"

    for file in files:
        match = re.match(pattern, file)
        if match:
            gcm, skenario, metode = match.groups()
            gcm_list.append(gcm)
            skenario_list.append(skenario)
            metode_list.append(metode)

    # Buat unik dan urut
    gcm_list = sorted(set(gcm_list))
    skenario_list = sorted(set(skenario_list))
    metode_list = sorted(set(metode_list))

    col1, col2, col3 = st.columns(3)

    # Select box untuk memilih
    with col1:
        gcm_selected = st.selectbox("Pilih GCM", gcm_list, key="rata1")
    with col2:
        skenario_selected = st.selectbox("Pilih Skenario", skenario_list, key="rata2")
    with col3:
        metode_selected = st.selectbox("Pilih Metode", metode_list, key="rata3")

    # Bangun nama file berdasarkan pilihan
    filename = f"{gcm_selected}_{skenario_selected}_2021_2100_{metode_selected}.nc"
    filepath = os.path.join(folder_path, filename)

    @st.cache_data
    def load_nc_file(path):
        return xr.open_dataset(path)

    data = load_nc_file(filepath)

    # Buat array waktu dari 2021 hingga 2100 (80 tahun)
    new_time = pd.date_range("2021-01-01", "2100-01-01", freq="YS")

    # Ganti koordinat waktu
    data = data.assign_coords(time=new_time)

    # Pastikan tipe data time adalah datetime
    data['time'] = data['time'].astype('datetime64[ns]')

    # (Opsional) Tambahkan encoding agar file disimpan dengan benar
    encoding = {"time": {"units": "days since 2021-01-01", "calendar": "standard"}}

    lat = data['latitude'].values
    lon = data['longitude'].values
    sla = data['zos']

    # Konversi waktu
    time = pd.to_datetime(data['time'].values)
    sla['time'] = time

    # Ambil rata-rata tahunan
    years = time.year
    unique_years = np.unique(years)
    min_year = years.min()
    max_year = years.max()

    # Slider untuk memilih tahun
    selected_year = st.slider('Pilih Tahun', min_value=int(min_year), max_value=int(max_year), value=int(min_year), step=1)
    # selected_year = st.slider('Pilih Tahun', unique_years)

    sla_yearly = sla.groupby('time.year').mean('time')
    sla_selected = sla_yearly.sel(year=selected_year)

    # Flatten data
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')
    sla_flat = sla_selected.values.flatten()
    lat_flat = lat_grid.flatten()
    lon_flat = lon_grid.flatten()

    valid_mask = ~np.isnan(sla_flat)
    lat_valid = lat_flat[valid_mask]
    lon_valid = lon_flat[valid_mask]
    sla_valid = sla_flat[valid_mask]

    fig_map = go.Figure()

    fig_map.add_trace(go.Scattermapbox(
        lat=lat_valid,
        lon=lon_valid,
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=4,
            color=sla_valid,
            colorscale='RdBu_r',
            cmin=-0.25,
            cmax=0.25,
            colorbar=dict(
                title='SLA (meter)',
                orientation='h',
                x=0.5,
                y=-0.15,
                len=0.8,
                thickness=15
            )
        ),
        text=[
            f'Lat: {lt:.2f}<br>Lon: {ln:.2f}<br>SLA: {z:.3f} m'
            for lt, ln, z in zip(lat_valid, lon_valid, sla_valid)
        ],
        hoverinfo='text'
    ))

    fig_map.update_layout(
        mapbox_style="carto-darkmatter",
        mapbox_center={"lat": -2, "lon": 118},
        mapbox_zoom=3.4,
        width=1000,
        height=480,
        margin={"r": 0, "t": 100, "l": 0, "b": 0},
        title={
            'text': f'Rata-Rata Tinggi Muka Laut Tahun {selected_year}',
            'x': 0.5, 'y': 0.9, 'xanchor': 'center', 'yanchor': 'top',
            'font': {'size': 20, 'family': 'Arial, sans-serif'}
        }
    )

    st.plotly_chart(fig_map)

    # =============================
    # 1. Input Koordinat Referensi
    # =============================
    st.subheader("Grafik Rata-rata Tinggi Muka Laut (TML) Per Titik")

    ref_lat = st.number_input("Masukkan Lintang (Latitude)", min_value=float(lat.min()), max_value=float(lat.max()), value=float(lat.mean()), key='number1')
    ref_lon = st.number_input("Masukkan Bujur (Longitude)", min_value=float(lon.min()), max_value=float(lon.max()), value=float(lon.mean()), key='number2')

    # Cari index grid terdekat
    nearest_lat_idx = np.abs(lat - ref_lat).argmin()
    nearest_lon_idx = np.abs(lon - ref_lon).argmin()
    nearest_lat = lat[nearest_lat_idx]
    nearest_lon = lon[nearest_lon_idx]

    # =============================
    # 2. Ambil Time Series dari Grid Terdekat
    # =============================
    zos_point = sla[:, nearest_lat_idx, nearest_lon_idx]

    # Debugging: Cek apakah semua nilai nan
    if np.all(np.isnan(zos_point)):
        st.warning(f"Tidak ada data TML di koordinat sekitar ({nearest_lat:.2f}°, {nearest_lon:.2f}°). Coba ubah titik koordinat.")
    else:
        df_point = pd.DataFrame({
            'time': time,
            'sla': zos_point.values
        })
        df_point['year'] = df_point['time'].dt.year

        # =============================
        # 3. Rata-rata Tahunan
        # =============================
        df_yearly = df_point.groupby('year')['sla'].mean().reset_index()

        # =============================
        # 4. Plot Line Chart
        # =============================
        fig_line = px.line(
            df_yearly,
            x='year',
            y='sla',
            markers=True,
            labels={'sla': 'TML (m)', 'year': 'Tahun'}
        )
        fig_line.update_layout(
            height=400,
            title={
                'text': f'Rata-rata Tahunan TML di Titik Terdekat ({nearest_lat:.2f}°, {nearest_lon:.2f}°)',
                'x': 0.5, 'y': 0.9, 'xanchor': 'center', 'yanchor': 'top',
                'font': {'size': 20, 'family': 'Arial, sans-serif'},
            },
            yaxis=dict(tickformat=".2f"),
        )

        st.plotly_chart(fig_line)

# ====================
# TAB 2: TREND SLA
# ====================
with tab2:
    # st.subheader("Peta Tren Proyeksi Tinggi Muka Laut Periode 2021-2100")

    # Jalankan fungsi
    with st.spinner("Sedang mengunduh dan mengekstrak data trend..."):
        folder_path1 = download_and_extract_zip(url1, output_folder1)

    # Folder trend
    TREND_FOLDER = folder_path1
    all_nc_files = [f for f in os.listdir(TREND_FOLDER) if f.endswith(".nc")]

    # Parsing metadata dari nama file
    file_info = []
    for f in all_nc_files:
        if f.startswith("SSH_trend_"):
            # Split file name based on underscores
            parts = f.replace(".nc", "").split("_")
            if len(parts) >= 6:  # Pastikan ada cukup bagian untuk diproses
                gcm = parts[2]  # Bagian GCM (misal: cmcc)
                skenario = parts[3]  # Bagian Skenario (misal: ssp245)
                metode = parts[-1] if parts[-1] in ["CNN", "CNN-LSTM"] else "unknown"  # Bagian Metode
                file_info.append({
                    "filename": f,
                    "gcm": gcm,
                    "skenario": skenario,
                    "metode": metode
                })

    # Pilihan unik
    gcm_options = sorted(set(i["gcm"] for i in file_info))
    skenario_options = sorted(set(i["skenario"] for i in file_info))
    metode_options = sorted(set(i["metode"] for i in file_info if i["metode"] != "unknown"))

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_gcm = st.selectbox("Pilih GCM", gcm_options, key="tren1")
    with col2:
        selected_skenario = st.selectbox("Pilih Skenario", skenario_options, key="tren2")
    with col3:
        selected_metode = st.selectbox("Pilih Metode", metode_options, key="tren3")

    # Cari file cocok
    selected_file = None
    for info in file_info:
        if (
            info["gcm"] == selected_gcm and
            info["skenario"] == selected_skenario and
            info["metode"] == selected_metode
        ):
            selected_file = os.path.join(TREND_FOLDER, info["filename"])
            break

    if selected_file:
        ds = xr.open_dataset(selected_file)
        trend = ds["trend"]
        lat = ds["latitude"].values
        lon = ds["longitude"].values
        trend_values = trend.values

        lon2d, lat2d = np.meshgrid(lon, lat)
        lat_flat = lat2d.flatten()
        lon_flat = lon2d.flatten()
        trend_flat = trend_values.flatten()

        valid_mask = ~np.isnan(trend_flat)
        lat_valid = lat_flat[valid_mask]
        lon_valid = lon_flat[valid_mask]
        trend_valid = trend_flat[valid_mask]

        fig_trend = go.Figure()

        fig_trend.add_trace(go.Scattermapbox(
            lat=lat_valid,
            lon=lon_valid,
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=4,
                color=trend_valid,
                colorscale='RdBu_r',
                cmin=-2,
                cmax=2,
                colorbar=dict(
                    title='Trend (mm/tahun)',
                    orientation='h',
                    x=0.5,
                    y=-0.15,
                    len=0.85,
                    thickness=15
                )
            ),
            text=[
                f"Lat: {lt:.2f}<br>Lon: {ln:.2f}<br>Trend: {tr:.2f} mm/yr"
                for lt, ln, tr in zip(lat_valid, lon_valid, trend_valid)
            ],
            hoverinfo='text'
        ))

        fig_trend.update_layout(
            mapbox_style="carto-darkmatter",
            mapbox_center={"lat": -2, "lon": 118},
            mapbox_zoom=3.4,
            width=1000,
            height=480,
            margin={"r": 0, "t": 100, "l": 0, "b": 0},
            title={'text':f"Tren TML Menggunakan {selected_gcm.upper()} - {selected_skenario.upper()} Metode {selected_metode.upper()} Periode 2021-2100",
                   'x': 0.5, 'y': 0.9, 'xanchor': 'center', 'yanchor': 'top',
                'font': {'size': 19, 'family': 'Arial, sans-serif'}},
        )

        st.plotly_chart(fig_trend)
    else:
        st.warning("File dengan kombinasi tersebut tidak ditemukan.")

    # 2 buat Grafik Trend
    st.subheader("Trend Rata-Rata Tinggi Muka Laut Periode 2021-2100")
    
    # Folder hasil prediksi
    folder_path = "hasil/"

    # Ambil semua nama file di folder
    files = [f for f in os.listdir(folder_path) if f.endswith(".nc")]

    # Ekstrak komponen: gcm, skenario, metode
    gcm_list = []
    skenario_list = []
    metode_list = []

    pattern = r"(.+?)_(ssp\d{3})_2021_2100_(.+?)\.nc"

    for file in files:
        match = re.match(pattern, file)
        if match:
            gcm, skenario, metode = match.groups()
            gcm_list.append(gcm)
            skenario_list.append(skenario)
            metode_list.append(metode)

    # Buat unik dan urut
    gcm_list = sorted(set(gcm_list))
    skenario_list = sorted(set(skenario_list))
    metode_list = sorted(set(metode_list))

    col1, col2, col3 = st.columns(3)

    # Select box untuk memilih
    with col1:
        gcm_selected = st.selectbox("Pilih GCM", gcm_list, key="grafik1")
    with col2:
        skenario_selected = st.selectbox("Pilih Skenario", skenario_list, key="grafik2")
    with col3:
        metode_selected = st.selectbox("Pilih Metode", metode_list, key="grafik3")

    # Bangun nama file berdasarkan pilihan
    filename = f"{gcm_selected}_{skenario_selected}_2021_2100_{metode_selected}.nc"
    filepath = os.path.join(folder_path, filename)

    data = load_nc_file(filepath)

    # Pastikan koordinat waktu benar
    new_time = pd.date_range("2021-01-01", "2100-01-01", freq="YS")
    data = data.assign_coords(time=new_time)
    data['time'] = data['time'].astype('datetime64[ns]')

    # Ambil variabel
    lat = data['latitude'].values
    lon = data['longitude'].values
    sla = data['zos']
    sla['time'] = pd.to_datetime(data['time'].values)

    # 1. Ambil rata-rata global untuk setiap waktu
    zos_global_avg = sla.mean(dim=['lat', 'lon'], skipna=True)

    # 2. Buat time series
    time_values = pd.to_datetime(zos_global_avg['time'].values)
    zos_values = zos_global_avg.values

    # 3. Konversi waktu ke angka (untuk regresi linier)
    x = np.arange(len(time_values))
    y = zos_values

    # 4. Hitung tren linier (slope dan intercept)
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    trend_line = intercept + slope * x

    # Konversi ke mm/tahun
    slope_mm_per_year = slope * 1000

    # 5. Plot time series + trend
    fig_trend = go.Figure()

    fig_trend.add_trace(go.Scatter(
        x=time_values,
        y=y,
        mode='lines',
        name='TML Per Tahun',
        line=dict(color='royalblue')
    ))

    fig_trend.add_trace(go.Scatter(
        x=time_values,
        y=trend_line,
        mode='lines',
        name=f'Trend (mm/tahun)',
        line=dict(color='firebrick', dash='dash')
    ))

    fig_trend.update_layout(
        title={'text':'Tren Tinggi Muka Laut (TML) Indonesia Periode 2021-2100',
               'x': 0.5, 'y': 0.9, 'xanchor': 'center', 'yanchor': 'top',
                'font': {'size': 20, 'family': 'Arial, sans-serif'}},
        xaxis_title='Time',
        yaxis=dict(title='TML (m)'), # range=[-0.0, 0.4]
        width=900,
        height=500,
        template='plotly_white',
        legend=dict(
            orientation='h',      # horizontal
            yanchor='top',        # anchor to the top of the legend box
            y=-0.3,               # place it below the x-axis
            xanchor='center',
            x=0.5
        )
    )

    fig_trend.add_annotation(
        x=time_values[-1],
        y=trend_line[-1],
        text=f"{slope_mm_per_year:.3f} mm/tahun",
        showarrow=False,
        font=dict(color='firebrick', size=12),
        xanchor='left'
    )

    st.plotly_chart(fig_trend)

# membuat narasi tabel dalam keterangan
with st.expander(":blue-background[Keterangan :]"):
    st.caption("**Penjelasan/Definisi**")
    st.caption(('''**Data proyeksi yang digunakan** data Multi-GCM dari ESGF dengan variabel zos (Sea Surface Height)
                dengan periode waktu bulanan (monthly) dengan 2 skenario yaitu ssp245 dan ssp585.'''))
    st.caption(('''Diproses dengan model deep learning CNN dan hybrid CNN-LSTM dengan proses tunning menggunakan
                Epoch 50 dan Batch Size 8 dengan EarlyStopping dari error 'loss'.'''))
    st.caption(('''Model dibangun dengan data inputan adalah GCM CMIP6 historis periode 1995-2014,
                dan data reanalysis dari CMEMS resolusi 0.083° x 0.083° (~9 km).
                Setelah model dilatih, kemudian model tersebut diberi inputan baru dari data future
                (data skenario iklim periode 2021 s.d 2100).'''))
