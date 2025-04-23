# import library
import streamlit as st

# Setting layout halaman
st.set_page_config(
        page_title="Analisis TML Indonesia",
        page_icon="🏠",
        layout="centered",
        initial_sidebar_state="expanded"
    )

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

######################################################
# Judul utama halaman
st.title('Analisis dan Proyeksi Tinggi Muka Laut di Indonesia')

# atribut tambahan
st.markdown('''
**Adityo Wicaksono**

*Badan Meteorologi Klimatologi dan Geofisika*

''')

st.divider()

# judul section
st.header('Overview')

# narasi utama (abstrak)
st.info('''
Perubahan iklim global telah menyebabkan peningkatan tinggi muka laut (TML) yang signifikan, terutama di wilayah pesisir negara kepulauan seperti Indonesia.
        Studi ini bertujuan untuk menganalisis tren historis serta memproyeksikan TML di wilayah Indonesia menggunakan pendekatan kecerdasan buatan berbasis Deep Learning.
        Model Deep Learning (CNN-LSTM) digunakan untuk menangkap dinamika spasial dan temporal dari data Sea Surface Height (SSH) yang diperoleh dari observasi satelit (CMEMS)
        dan proyeksi model iklim (GCM CMIP6) pada skenario SSP245 dan SSP585.

Data SSH diolah melalui serangkaian tahapan mulai dari slicing, normalisasi, hingga pembentukan sekuens waktu untuk input ke dalam arsitektur model.
        Model dilatih menggunakan data historis (1995-2014) dan dievaluasi untuk memproyeksikan kondisi masa depan (2021-2100).
        Hasil proyeksi menunjukkan adanya kecenderungan peningkatan TML yang konsisten pada seluruh skenario pada sebagian besar GCM, dengan variasi spasial yang signifikan antar wilayah.

Temuan ini diharapkan dapat memberikan kontribusi terhadap pemahaman dinamika tinggi muka laut di Indonesia dan mendukung perencanaan adaptasi serta mitigasi risiko perubahan iklim di wilayah pesisir.
''')

# atribut tambahan
st.markdown('**Keywords:** *Tinggi Muka Laut*, *Deep Learning*, *Indonesia*, *Proyeksi Iklim*')

# judul section 1
st.header('Latar Belakang')

# narasi pendahuluan
st.markdown('''
            <div class="justified-text">
Perubahan iklim global berdampak signifikan terhadap kenaikan tinggi muka laut di berbagai wilayah dunia,
            termasuk Indonesia yang merupakan negara kepulauan. Peningkatan suhu global menyebabkan pencairan es kutub dan
            ekspansi termal air laut yang secara langsung memicu kenaikan tinggi muka laut (sea level rise, SLR).
            Kondisi ini menjadi ancaman serius terhadap ekosistem pesisir, infrastruktur, dan masyarakat yang tinggal di wilayah pesisir Indonesia.  
            </div>   
''', unsafe_allow_html=True)
st.markdown(''' ''')
st.markdown('''
            <div class="justified-text">
Untuk memahami dan memproyeksikan dampak jangka panjang dari fenomena ini, diperlukan pendekatan ilmiah berbasis data yang mampu menangkap kompleksitas spasial
            dan temporal dari perubahan tinggi muka laut. Oleh karena itu, studi ini mengusulkan penggunaan kombinasi data observasi dan proyeksi model iklim
            dengan pendekatan Deep Learning, khususnya model CNN-LSTM yang telah terbukti efektif dalam menangani data spasio-temporal.
            </div>   
''', unsafe_allow_html=True)
st.markdown(''' ''')

# judul section 2
st.header('Data dan Metode')

# narasi section 2
st.markdown('''
            <div class="justified-text">
Penelitian ini menggunakan data Sea Surface Height (SSH) observasi dari CMEMS (Copernicus Marine) dengan resolusi spasial 0.083°x0.083°
            dan data Sea Surface Height proyeksi dari model iklim GCM CMIP6 yang terbagi menjadi 2 resolusi, yaitu resolusi
            tinggi (< 1°x1°) seperti ACCESS, ACCESS-CM2, CANESM55, dan CMCC. Sedangkan GCM CMIP6 dengan resolusi rendah
            (> 1°x1°) seperti EC-EARTH, MPI, dan MIROC6.
            Data dikumpulkan dari rentang waktu 1995 hingga 2014 untuk data historis dan dari tahun 2021 hingga 2100
            untuk proyeksi masa depan (SSP245 dan SSP585).
            </div>
''', unsafe_allow_html=True)
st.markdown(''' ''')
st.markdown('''
            <div class="justified-text">
Data GCM dan SSH terlebih dahulu dipotong (slicing) berdasarkan wilayah Indonesia (90°-145° BT dan -15°-10° LS).
            Selanjutnya dilakukan pre-processing berupa:
            </div>
''', unsafe_allow_html=True)
cola, colb = st.columns([1,30], gap="small")
with cola:
    st.markdown('''**a.**''')
with colb:
    st.markdown('''
            <div class="justified-text">
Penyamaan ukuran dan rentang waktu</div>''', unsafe_allow_html=True)
cola, colb = st.columns([1,30], gap="small")
with cola:
    st.markdown('''**b.**''')
with colb:
    st.markdown('''
            <div class="justified-text">
Pengisian nilai hilang</div>''', unsafe_allow_html=True)
cola, colb = st.columns([1,30], gap="small")
with cola:
    st.markdown('''**c.**''')
with colb:
    st.markdown('''
            <div class="justified-text">
Normalisasi data menggunakan teknik Min-Max Scaling</div>''', unsafe_allow_html=True)
cola, colb = st.columns([1,30], gap="small")
with cola:
    st.markdown('''**d.**''')
with colb:
    st.markdown('''
            <div class="justified-text">
Pembuatan sekuens data untuk input CNN-LSTM</div>''', unsafe_allow_html=True)

# gambar alur
st.image("alur load.png", caption="Pre-Processing Data GCM")

st.markdown('''
            <div class="justified-text">
Penelitian ini menggunakan pendekatan berbasis Deep Learning, khususnya kombinasi arsitektur Convolutional Neural Network (CNN)
            dan Long Short-Term Memory (LSTM), model dijalankan dalam dua metode, pertama single CNN, dan kedua metode hybrid CNN dan LSTM.
            Metode ini dipilih karena kemampuannya dalam menangani data yang memiliki dimensi spasial dan temporal secara simultan,
            seperti data tinggi muka laut (Sea Surface Height/SSH) dari satelit dan model iklim.
            </div>
''', unsafe_allow_html=True)
st.markdown(''' ''')
st.markdown('''
            <div class="justified-text">
Secara umum, setiap input SSH (Sea Surface Height) akan diproses oleh CNN untuk mendapatkan representasi spasialnya,
            lalu serangkaian representasi ini dimasukkan ke dalam LSTM untuk mempelajari perubahan pola dari waktu ke waktu.
            Dalam proses pembelajaran pola tersebut model CNN atau CNN-LSTM juga akan berusaha meningkatkan resolusi spasial
            dari data inputan GCM CMIP6 mengikuti resolusi spasial dari data Observasinya (CMEMS) atau biasa disebut dengan Spatial Downscaling.
            Arsitektur ini memungkinkan model untuk memprediksi kondisi TML masa depan berdasarkan dinamika masa lalu.
            </div>
''', unsafe_allow_html=True)

st.image("alur model cnn-lstm.png", caption="Proses Normalisasi hingga Arsitektur Model")

# judul section 3
st.header('Manfaat')

# narasi section 3
st.markdown('''**Penelitian ini diharapkan dapat memberikan kontribusi sebagai berikut:**''')
st.markdown(''' ''')
col1, col2 = st.columns([1,30], gap="small")
with col1:
    st.markdown('''**1.**''')
with col2:
    st.markdown('''
                <div class="justified-text">
    Memberikan proyeksi jangka panjang tinggi muka laut di Indonesia berdasarkan skenario iklim masa depan.</div> 
    ''', unsafe_allow_html=True)
col1, col2 = st.columns([1,30], gap="small")
with col1:
    st.markdown('''**2.**''')
with col2:
    st.markdown('''
                <div class="justified-text">
    Menawarkan pendekatan model prediktif berbasis Deep Learning untuk analisis iklim dengan akurasi tinggi.</div> 
    ''', unsafe_allow_html=True)
col1, col2 = st.columns([1,30], gap="small")
with col1:
    st.markdown('''**3.**''')
with col2:
    st.markdown('''
                <div class="justified-text">
    Menjadi sumber informasi penting untuk pengambilan keputusan dalam mitigasi dan adaptasi terhadap dampak perubahan iklim,
                terutama di wilayah pesisir.</div> 
    ''', unsafe_allow_html=True)
col1, col2 = st.columns([1,30], gap="small")
with col1:
    st.markdown('''**4.**''')
with col2:
    st.markdown('''
                <div class="justified-text">
    Mendorong pemanfaatan data iklim dan teknik komputasi modern dalam perencanaan pembangunan berkelanjutan.</div> 
    ''', unsafe_allow_html=True)
