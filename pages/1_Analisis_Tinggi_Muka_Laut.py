import streamlit as st
import xarray as xr
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import zipfile
import os
import requests
import io
from matplotlib import cm
from matplotlib.colors import to_hex

# Setting layout halaman
st.set_page_config(
        page_title="Analisis TML Indonesia",
        page_icon="🏠",
        layout="centered",
        initial_sidebar_state="expanded"
    )

# # Load custom CSS file
# with open('style.css') as f:
#     st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

@st.cache_resource
def download_and_open_nc():
    url = "https://github.com/adityoAJA/SLR/releases/download/v1/cmems_obs.zip"

    nc_filename = "cmems_obs.nc"
    if not os.path.exists(nc_filename):
        response = requests.get(url)
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(".")  # Ekstrak ke current folder

    return nc_filename

# judul section
st.title('Analisis Tinggi Muka Laut')

# st.divider()

# BAGI 3 TAB
tab1, tab2, tab3 = st.tabs(["🔍 SLA Tahunan", "📈 Tren Time Series", "🗺 Peta Tren SLA"])

with tab1:
    # 1. buat peta rata2 tahunan
    # Panggil fungsi
    with st.spinner("Sedang mengunduh dan membuka NetCDF..."):
        ds = download_and_open_nc()

    @st.cache_data
    def load_nc_file(file_path):
        return xr.open_dataset(file_path)

    # load data
    data = load_nc_file(ds)

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

    # # Filter data valid
    valid_mask = ~np.isnan(zos_flat)
    lat_valid = lat_flat[valid_mask]
    lon_valid = lon_flat[valid_mask]
    zos_valid = zos_flat[valid_mask]

    # 1. Bikin bins dan warna diskrit
    zmin, zmax = -0.25, 0.25
    bins = np.linspace(zmin, zmax, 11)  # 9 kelas

    # Bikin palet dari cmap
    cmap = cm.get_cmap('coolwarm', 10)
    colors = [to_hex(cmap(i / (cmap.N - 1))) for i in range(cmap.N)]

    # Tambahkan warna ekstrem di ujung yang benar-benar beda
    color_min = "#081d58"      # Warna lebih ekstrem kiri
    color_max = "#660000"     # Warna lebih ekstrem kanan

    # Gabungkan warna
    colors_ext = [color_min] + colors + [color_max]

    # 2. Diskritisasi data ke index bin
    bin_indices = np.digitize(zos_valid, bins)
    bin_indices = np.clip(bin_indices, 0, len(colors))  # dari 0 sampai len(colors)

    # Membuat peta
    fig_map = go.Figure()

    # 4. Tambahkan satu trace untuk setiap kelas (tanpa legend)
    for i in range(len(colors_ext)):
        mask = bin_indices == i
        if np.sum(mask) == 0:
            continue
        fig_map.add_trace(go.Scattermapbox(
            lat=np.array(lat_valid)[mask],
            lon=np.array(lon_valid)[mask],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=3.3,
                color=colors_ext[i],
            ),
            showlegend=False,
            hoverinfo='text',
            text=[
                f"Lintang: {lt:.2f}<br>Bujur: {ln:.2f}<br>TML: {z:.3f} m"
                for lt, ln, z in zip(np.array(lat_valid)[mask],
                                    np.array(lon_valid)[mask],
                                    np.array(zos_valid)[mask])
            ]
        ))
    
    # ==================== Bar Colorbar Horizontal ==================== #
    bar_x = list(range(len(colors_ext)))  # agar bisa pakai tickvals & ticktext
    bar_y = [2] * len(colors_ext)  # tetap 1 karena nanti domainnya kecil

    fig_map.add_trace(go.Bar(
        x=bar_x,
        y=bar_y,
        marker=dict(color=colors_ext),
        width=0.95,  # agar tiap bar mepet, tidak ada spasi
        orientation='v',
        showlegend=False,
        hoverinfo="none",
        xaxis='x2',
        yaxis='y2'
    ))

    # ==================== Anotasi Label dan Judul Colorbar ==================== #
    center_lat = -2
    center_lon = 118

    fig_map.update_layout(
        mapbox_style="carto-darkmatter",
        mapbox_zoom=3.3,
        autosize=True,
        width=2000,
        height=550,
        mapbox_center={"lat": center_lat, "lon": center_lon},
        
        # Geser mapbox naik
        mapbox=dict(domain={'y': [0.1, 1]}),
        
        margin=dict(l=0, r=0, t=90, b=100),
        title={'text': f'Peta Rata-Rata Tinggi Muka Laut (TML) Tahun {selected_year}',
            'x': 0.5, 'y': 0.9, 'xanchor': 'center', 'yanchor': 'top',
            'font': {'size': 20, 'family': 'Arial, sans-serif'}},

        # Sumbu untuk colorbar horizontal
        xaxis2=dict(
            domain=[0.15, 0.85],
            anchor='y2',
            tickmode='array',
            tickvals = list(range(len(colors_ext))),
            ticktext = ["< {:.2f}".format(bins[0])] + \
                ["{:.2f} – {:.2f}".format(bins[i], bins[i+1]) for i in range(len(bins)-1)] + \
                ["> {:.2f}".format(bins[-1])],
            tickangle=45,  # biar teks horizontal (tidak miring)
            tickfont=dict(size=11),  # kecilkan ukuran teks
            showline=False,
            showgrid=False,
            zeroline=False
        ),
        yaxis2=dict(
            domain=[0.02, 0.08],
            anchor='x2',
            showticklabels=False,
            showgrid=False,
            zeroline=False
        ),

        # Label bawah
        annotations=[
            dict(
                x=0.5, y=-0.25, xref="paper", yref="paper",
                text="Laju Perubahan (m/tahun)",
                showarrow=False, font=dict(size=12, color="black")
            )
        ]
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

    # # Skala warna dan range
    # colorscale = 'RdBu_r'
    # zmin = -5  # Sesuaikan dengan range data kamu
    # zmax = 5

    # 1. Bikin bins dan warna diskrit
    zmin, zmax = -5, 5
    bins = np.linspace(zmin, zmax, 11)  # 9 kelas

    # Bikin palet dari cmap
    cmap = cm.get_cmap('coolwarm', 10)
    colors = [to_hex(cmap(i / (cmap.N - 1))) for i in range(cmap.N)]

    # Tambahkan warna ekstrem di ujung yang benar-benar beda
    color_min = "#081d58"      # Warna lebih ekstrem kiri
    color_max = "#660000"     # Warna lebih ekstrem kanan

    # Gabungkan warna
    colors_ext = [color_min] + colors + [color_max]

    # 2. Diskritisasi data ke index bin
    bin_indices = np.digitize(trend_valid, bins)
    bin_indices = np.clip(bin_indices, 0, len(colors))  # dari 0 sampai len(colors)

    # Membuat peta
    fig_map_trend = go.Figure()

    # 4. Tambahkan satu trace untuk setiap kelas (tanpa legend)
    for i in range(len(colors_ext)):
        mask = bin_indices == i
        if np.sum(mask) == 0:
            continue
        fig_map_trend.add_trace(go.Scattermapbox(
            lat=np.array(lat_valid)[mask],
            lon=np.array(lon_valid)[mask],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=3.3,
                color=colors_ext[i],
            ),
            showlegend=False,
            hoverinfo='text',
            text=[
                f"Lintang: {lt:.2f}<br>Bujur: {ln:.2f}<br>TML: {z:.3f} m"
                for lt, ln, z in zip(np.array(lat_valid)[mask],
                                    np.array(lon_valid)[mask],
                                    np.array(trend_valid)[mask])
            ]
        ))
    
    # ==================== Bar Colorbar Horizontal ==================== #
    bar_x = list(range(len(colors_ext)))  # agar bisa pakai tickvals & ticktext
    bar_y = [2] * len(colors_ext)  # tetap 1 karena nanti domainnya kecil

    fig_map_trend.add_trace(go.Bar(
        x=bar_x,
        y=bar_y,
        marker=dict(color=colors_ext),
        width=0.95,  # agar tiap bar mepet, tidak ada spasi
        orientation='v',
        showlegend=False,
        hoverinfo="none",
        xaxis='x2',
        yaxis='y2'
    ))

    # ==================== Anotasi Label dan Judul Colorbar ==================== #
    center_lat = -2
    center_lon = 118

    fig_map_trend.update_layout(
        mapbox_style="carto-darkmatter",
        mapbox_zoom=3.3,
        autosize=True,
        width=2000,
        height=550,
        mapbox_center={"lat": center_lat, "lon": center_lon},
        
        # Geser mapbox naik
        mapbox=dict(domain={'y': [0.1, 1]}),
        
        margin=dict(l=0, r=0, t=90, b=100),
        title={'text':"Peta Tren Tinggi Muka Laut Periode 1993-2023",
               'x': 0.5, 'y': 0.9, 'xanchor': 'center', 'yanchor': 'top',
                'font': {'size': 20, 'family': 'Arial, sans-serif'}},

        # Sumbu untuk colorbar horizontal
        xaxis2=dict(
            domain=[0.15, 0.85],
            anchor='y2',
            tickmode='array',
            tickvals = list(range(len(colors_ext))),
            ticktext = ["< {:.2f}".format(bins[0])] + \
                ["{:.2f} – {:.2f}".format(bins[i], bins[i+1]) for i in range(len(bins)-1)] + \
                ["> {:.2f}".format(bins[-1])],
            tickangle=45,  # biar teks horizontal (tidak miring)
            tickfont=dict(size=11),  # kecilkan ukuran teks
            showline=False,
            showgrid=False,
            zeroline=False
        ),
        yaxis2=dict(
            domain=[0.02, 0.08],
            anchor='x2',
            showticklabels=False,
            showgrid=False,
            zeroline=False
        ),

        # Label bawah
        annotations=[
            dict(
                x=0.5, y=-0.25, xref="paper", yref="paper",
                text="Laju Perubahan (m/tahun)",
                showarrow=False, font=dict(size=12, color="black")
            )
        ]
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
