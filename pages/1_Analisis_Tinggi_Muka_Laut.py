import streamlit as st
import xarray as xr
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import io
import tempfile
import os
import requests

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

@st.cache_resource
def download_and_open_nc():
    url = "https://github.com/adityoAJA/SLR/releases/download/v1/cmems_obs.nc"
    
    # Simpan di file temporer dengan suffix .nc
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nc") as tmp_file:
        temp_path = tmp_file.name

        # Download dengan stream
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=8192):
                tmp_file.write(chunk)

    # Coba buka file NetCDF
    try:
        ds = xr.open_dataset(temp_path, engine="netcdf4")  # bisa coba 'h5netcdf' juga kalau error
        return ds
    except Exception as e:
        st.error(f"Gagal membuka NetCDF: {e}")
        return None

# judul section
st.title('Analisis Tinggi Muka Laut')

# st.divider()

# BAGI 3 TAB
tab1, tab2, tab3 = st.tabs(["📍 SLA Tahunan", "📈 Tren Time Series", "🌊 Peta Tren SLA"])

with tab1:
    # 1. buat peta rata2 tahunan
    # Panggil fungsi
    with st.spinner("Sedang mengunduh dan membuka NetCDF..."):
        ds = download_and_open_nc()

    data = ds

    # Variabel
    lat = data['latitude'].values
    lon = data['longitude'].values
    zos = data['sla']

    # Konversi waktu
    time = pd.to_datetime(data['time'].values)
    zos['time'] = time

    # Ambil tahun dari waktu
    years = time.year
    unique_years = np.unique(years)

    # Pilihan tahun
    selected_year = st.selectbox('Pilih Tahun', unique_years)

    # Ambil data rata-rata tahunan
    zos_yearly = zos.groupby('time.year').mean('time')
    zos_selected = zos_yearly.sel(year=selected_year)

    # # Konversi ke mm/year
    # zos_selected = zos_selected * 1000  # dari meter ke mm

    # Flatten data
    lat_flat = np.repeat(lat, len(lon))
    lon_flat = np.tile(lon, len(lat))
    zos_flat = zos_selected.values.flatten()

    # Filter data valid
    valid_mask = ~np.isnan(zos_flat)
    lat_valid = lat_flat[valid_mask]
    lon_valid = lon_flat[valid_mask]
    zos_valid = zos_flat[valid_mask]

    # Pewarnaan dengan skala divergen
    colorscale = 'RdBu_r'
    zmin = -0.25  # sesuaikan dengan data kamu
    zmax = 0.25

    # Membuat peta
    fig_map = go.Figure()

    fig_map.add_trace(go.Scattermapbox(
        lat=lat_valid,
        lon=lon_valid,
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=4,
            color=zos_valid,
            colorscale=colorscale,
            cmin=zmin,
            cmax=zmax,
            colorbar=dict(
                title='SLA (meter)',
                orientation='h',
                x=0.5,
                y=-0.15,
                len=0.8,
                thickness=15
            )
        ),
        text=[f'Lintang: {lt:.2f}<br>Bujur: {ln:.2f}<br>TML: {z:.3f} m' 
            for lt, ln, z in zip(lat_valid, lon_valid, zos_valid)],
        hoverinfo='text'
    ))

    # Pusat peta
    center_lat = -2
    center_lon = 118

    # Layout peta
    fig_map.update_layout(
        mapbox=dict(
            style="carto-darkmatter", # "carto-positron" atau "carto-darkmatter"
            center={"lat": center_lat, "lon": center_lon},
            zoom=3.4,
        ),
        width=1000,
        height=480,
        title={'text': f'Peta Rata-Rata Tinggi Muka Laut (TML) Tahun {selected_year}',
            'x': 0.5, 'y': 0.9, 'xanchor': 'center', 'yanchor': 'top',
            'font': {'size': 20, 'family': 'Arial, sans-serif'}},
        autosize=True,
        margin={"r": 0, "t": 100, "l": 0, "b": 0}
    )

    st.plotly_chart(fig_map)

    # =============================
    # 1. Input Koordinat Referensi
    # =============================
    st.subheader("Grafik Rata-rata TML Per Titik")

    ref_lat = st.number_input("Masukkan Lintang (Latitude)", min_value=float(lat.min()), max_value=float(lat.max()), value=float(lat.mean()))
    ref_lon = st.number_input("Masukkan Bujur (Longitude)", min_value=float(lon.min()), max_value=float(lon.max()), value=float(lon.mean()))

    # Cari index grid terdekat
    nearest_lat_idx = np.abs(lat - ref_lat).argmin()
    nearest_lon_idx = np.abs(lon - ref_lon).argmin()
    nearest_lat = lat[nearest_lat_idx]
    nearest_lon = lon[nearest_lon_idx]

    # =============================
    # 2. Ambil Time Series dari Grid Terdekat
    # =============================
    zos_point = zos[:, nearest_lat_idx, nearest_lon_idx]

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
                'font': {'size': 20, 'family': 'Arial, sans-serif'}
            },
            yaxis=dict(tickformat=".2f"),
        )

        st.plotly_chart(fig_line)

with tab2:
    # 2 buat Grafik Trend
    from scipy.stats import linregress

    # 1. Ambil rata-rata global untuk setiap waktu
    zos_global_avg = zos.mean(dim=['latitude', 'longitude'], skipna=True)

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
    slope_mm_per_year = slope * 1000 * 12

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
        title={'text':'Tren Tinggi Muka Laut (TML) Indonesia Periode 1993-2023',
               'x': 0.5, 'y': 0.9, 'xanchor': 'center', 'yanchor': 'top',
                'font': {'size': 20, 'family': 'Arial, sans-serif'}},
        xaxis_title='Time',
        yaxis_title='TML (m)',
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
        text=f"{slope_mm_per_year:.2f} mm/tahun",
        showarrow=False,
        font=dict(color='firebrick', size=12),
        xanchor='left'
    )

    st.plotly_chart(fig_trend)

with tab3:
    # 3. peta tren spasial
    # Baca file NetCDF yang sudah dihitung tren (zos_trend_annual_mm_per_year.nc)
    @st.cache_data
    def load_trend_data(file_path):
        ds1 = xr.open_dataset(file_path)
        return ds1

    # Load data tren
    trend_ds = load_trend_data("SSH_trend_indo_1993_2014.nc")

    trend = trend_ds["trend"]

    # Ambil koordinat
    lat = trend_ds["latitude"].values
    lon = trend_ds["longitude"].values
    trend_values = trend.values

    # Buat meshgrid untuk koordinat
    lon2d, lat2d = np.meshgrid(lon, lat)

    # Flatten
    lat_flat = lat2d.flatten()
    lon_flat = lon2d.flatten()
    trend_flat = trend_values.flatten()

    # Filter NaN
    valid_mask = ~np.isnan(trend_flat)
    lat_valid = lat_flat[valid_mask]
    lon_valid = lon_flat[valid_mask]
    trend_valid = trend_flat[valid_mask]

    # Skala warna dan range
    colorscale = 'RdBu_r'
    zmin = -5  # Sesuaikan dengan range data kamu
    zmax = 5

    # Buat peta dengan plotly go
    fig_map_trend = go.Figure()

    fig_map_trend.add_trace(go.Scattermapbox(
        lat=lat_valid,
        lon=lon_valid,
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=4,
            color=trend_valid,
            colorscale=colorscale,
            cmin=zmin,
            cmax=zmax,
            colorbar=dict(
                title='Trend (mm/tahun)',
                orientation='h',
                x=0.5,
                y=-0.15,
                len=0.85,
                thickness=15
            )
        ),
        text=[f"Lat: {lt:.2f}<br>Lon: {ln:.2f}<br>Trend: {tr:.2f} mm/yr"
            for lt, ln, tr in zip(lat_valid, lon_valid, trend_valid)],
        hoverinfo='text'
    ))

    fig_map_trend.update_layout(
        mapbox_style="carto-darkmatter",
        mapbox_zoom=3.4,
        mapbox_center={"lat": -2, "lon": 118},
        width=1000,
        height=480,
        margin={"r":0, "t":100, "l":0, "b":0},
        title={'text':"Peta Tren Sea Surface Height Periode 1993-2023",
               'x': 0.5, 'y': 0.9, 'xanchor': 'center', 'yanchor': 'top',
                'font': {'size': 20, 'family': 'Arial, sans-serif'}},
    )

    st.plotly_chart(fig_map_trend)

# membuat narasi tabel dalam keterangan
with st.expander(":blue-background[Keterangan :]"):
    st.caption("**Penjelasan/Definisi**")
    st.caption(('''**Data observasi yang digunakan** dari Copernicus Marine Environment Monitoring Service (CMEMS)
                terkait sea level anomaly (SLA) Resolusi spasial adalah 0.125° x 0.125° (~12-13 km).'''))
    st.caption(('''Data ini merupakan Multi-Year reprocessed data dari semua **satelit altimeter** yang tersedia
                (seperti TOPEX, Jason, Envisat, CryoSat, dll) dan diproses menggunakan sistem DUACS
                (Data Unification and Altimeter Combination System).'''))
    st.caption(('''Frekuensi temporal adalah **bulanan (monthly)** dengan metode penggabungan (mean)
                dengan Periode data dari Januari 1993 sampai Desember 2023.'''))
