# import library
import streamlit as st

# Setting layout halaman
st.set_page_config(
        page_title="SLA-Indonesia",
        page_icon="🏠",
        layout="centered",
        initial_sidebar_state="expanded"
    )

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

######################################################
# Judul utama halaman
st.title('Analisis dan Proyeksi Muka Air Laut di Indonesia')

# atribut tambahan
st.markdown('''
**Badan Meteorologi, Klimatologi, dan Geofisika Indonesia**

''')

st.divider()

# judul section
st.header('Overview')

# narasi utama (abstract)
st.info('''
Perubahan iklim global telah menyebabkan kenaikan muka air laut (SLA) yang signifikan, terutama di wilayah pesisir negara kepulauan seperti Indonesia.
Penelitian ini bertujuan untuk menganalisis tren historis dan proyeksi SLA di wilayah Indonesia menggunakan pendekatan kecerdasan buatan berbasis Deep Learning.
Model Deep Learning (CNN-LSTM) digunakan untuk menangkap dinamika spasial dan temporal data Tinggi Permukaan Laut (SSH) yang diperoleh dari pengamatan satelit (CMEMS) dan proyeksi model iklim (GCM CMIP6) dalam skenario SSP245 dan SSP585.

Data SSH diolah melalui serangkaian tahapan mulai dari slicing, normalisasi, hingga pembentukan time sequence untuk input ke dalam arsitektur model.
Model dilatih menggunakan data historis (1995-2014) dan dievaluasi untuk memproyeksikan kondisi masa depan (2021-2100).
Hasil proyeksi menunjukkan tren peningkatan TML yang konsisten di semua skenario di sebagian besar GCM, dengan variasi spasial yang signifikan antar wilayah.

Temuan ini diharapkan dapat berkontribusi pada pemahaman dinamika permukaan laut di Indonesia dan mendukung perencanaan adaptasi dan mitigasi risiko perubahan iklim di wilayah pesisir.
''')

# atribut tambahan
st.markdown('**Keywords:** *Muka Air Laut*, *Deep Learning*, *Indonesia*, *Proyeksi Iklim*')

# judul section 1
st.header('Latar Belakang')

# narasi pendahuluan
st.markdown('''
            <div class="justified-text">
Perubahan iklim global berdampak signifikan terhadap kenaikan muka air laut di berbagai wilayah di dunia, termasuk Indonesia yang merupakan negara kepulauan.
Peningkatan suhu global tersebut menyebabkan mencairnya es di kutub dan pemuaian termal air laut yang secara langsung memicu terjadinya analisis muka air laut (SLA).
Kondisi ini menimbulkan ancaman serius terhadap ekosistem pesisir, infrastruktur, dan masyarakat yang tinggal di wilayah pesisir Indonesia. 
            </div>   
''', unsafe_allow_html=True)
st.markdown(''' ''')
st.markdown('''
            <div class="justified-text">
Untuk memahami dan memproyeksikan dampak jangka panjang dari fenomena ini, diperlukan pendekatan ilmiah berbasis data yang mampu menangkap kompleksitas spasial dan temporal dari perubahan muka air laut.
Oleh karena itu, penelitian ini mengusulkan penggunaan kombinasi data observasi dan proyeksi model iklim dengan pendekatan Deep Learning, khususnya model CNN yang telah terbukti efektif dalam menangani data spasial.
            </div>   
''', unsafe_allow_html=True)
st.markdown(''' ''')

# judul section 2
st.header('Data dan Metode')

# narasi section 2
st.markdown('''
            <div class="justified-text">
Penelitian ini menggunakan data observasi Ketinggian Permukaan Laut (SSH) dari CMEMS (Copernicus Marine) dengan resolusi spasial 0,083°x0,083°
dan data proyeksi Ketinggian Permukaan Laut dari model iklim CMIP6 GCM seperti ACCESS, ACCESS-CM2, CANESM55, CMCC, EC-EARTH, MPI, dan MIROC6.
Data dikumpulkan dari rentang waktu 1995 hingga 2014 untuk data historis dan dari 2021 hingga 2100
untuk proyeksi masa depan (SSP245 dan SSP585).
            </div>
''', unsafe_allow_html=True)
st.markdown(''' ''')
st.markdown('''
            <div class="justified-text">
Data GCM dan SSH terlebih dahulu diiris berdasarkan wilayah Indonesia (90°-145° BT dan -15°-10° LS).
Selanjutnya dilakukan pra-pemrosesan berupa:
            </div>
''', unsafe_allow_html=True)
cola, colb = st.columns([1,30], gap="small")
with cola:
    st.markdown('''**a.**''')
with colb:
    st.markdown('''
            <div class="justified-text">
Penyetaraan domain wilayah dan time series</div>''', unsafe_allow_html=True)
cola, colb = st.columns([1,30], gap="small")
with cola:
    st.markdown('''**b.**''')
with colb:
    st.markdown('''
            <div class="justified-text">
Mengisi nilai hilang</div>''', unsafe_allow_html=True)
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
Pembuatan urutan data pelatihan dan pengujian untuk input model CNN</div>''', unsafe_allow_html=True)

# gambar alur
st.image("alur load.png", caption="Pra-Pemrosesan GCM dan Data Observasi")

st.markdown('''
            <div class="justified-text">
Penelitian ini menggunakan pendekatan berbasis Deep Learning, khususnya arsitektur Convolutional Neural Network (CNN) dan juga mencoba menggunakan gabungan Convolutional Neural Network (CNN)
dan Long Short-Term Memory (LSTM), model dijalankan dengan dua metode, yaitu metode single CNN dan metode hybrid CNN dan LSTM.
Metode ini dipilih karena kemampuannya dalam menangani data yang berdimensi spasial dan temporal secara bersamaan, seperti data tinggi muka laut (SSH) dari satelit dan model iklim.
            </div>
''', unsafe_allow_html=True)
st.markdown(''' ''')
st.markdown('''
            <div class="justified-text">
Secara umum, setiap input SSH (Sea Surface Height) akan diproses oleh CNN untuk mendapatkan representasi spasialnya,
kemudian serangkaian representasi tersebut dimasukkan ke dalam LSTM untuk mempelajari perubahan pola dari waktu ke waktu. Dalam proses pembelajaran pola tersebut,
model CNN atau CNN-LSTM juga akan mencoba meningkatkan resolusi spasial data input GCM CMIP6 mengikuti resolusi spasial data Observasinya (CMEMS) atau yang biasa disebut Downscaling.
Arsitektur ini memungkinkan model untuk memprediksi kondisi SLR di masa mendatang berdasarkan dinamika masa lalu.
            </div>
''', unsafe_allow_html=True)

st.image("alur model cnn-lstm.png", caption="Arsitektur Model")

# judul section 3
st.header('Manfaat')

# narasi section 3
st.markdown('''**Penelitian ini diharapkan memberikan kontribusi sebagai berikut:**''')
st.markdown(''' ''')
col1, col2 = st.columns([1,30], gap="small")
with col1:
    st.markdown('''**1.**''')
with col2:
    st.markdown('''
                <div class="justified-text">
    Menyediakan proyeksi jangka panjang permukaan laut di Indonesia berdasarkan skenario iklim masa depan.</div> 
    ''', unsafe_allow_html=True)
col1, col2 = st.columns([1,30], gap="small")
with col1:
    st.markdown('''**2.**''')
with col2:
    st.markdown('''
                <div class="justified-text">
    Menawarkan pendekatan pemodelan prediktif berbasis Pembelajaran Mendalam untuk analisis iklim dengan akurasi tinggi.</div> 
    ''', unsafe_allow_html=True)
col1, col2 = st.columns([1,30], gap="small")
with col1:
    st.markdown('''**3.**''')
with col2:
    st.markdown('''
                <div class="justified-text">
    Menjadi sumber informasi penting bagi pengambilan keputusan dalam upaya mitigasi dan adaptasi terhadap dampak perubahan iklim, khususnya di wilayah pesisir.</div> 
    ''', unsafe_allow_html=True)
col1, col2 = st.columns([1,30], gap="small")
with col1:
    st.markdown('''**4.**''')
with col2:
    st.markdown('''
                <div class="justified-text">
    Mempromosikan penggunaan data iklim dan teknik komputasi modern dalam perencanaan pembangunan berkelanjutan.</div> 
    ''', unsafe_allow_html=True)
