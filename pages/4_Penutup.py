# import library
import streamlit as st

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

# judul section
st.title('Penutup')

st.divider()

# judul section
st.header('1. Kesimpulan')

# narasi utama
st.markdown('''
            <div class="justified-text">
Hasil evaluasi model CNN dan CNN-LSTM menunjukkan bahwa kedua model itu secara umum dapat meningkatkan Korelasi dan juga
            menurunkan nilai error dari setiap GCM. Namun model CNN terlihat cukup unggul dibandingkan dengan model CNN-LSTM
            dalam hasil evaluasi metric setiap GCM. Hal tersebut bisa jadi karena pola temporal dalam GCM tidak terlalu signifikan
            dalam proses model tersebut, justru layer-layer dan encoder konvolusi dalam CNN lah yang cukup berperan penting
            dalam membangun proses model khususnya dalam meningkatkan resolusi spasial dari GCM tadi.
            </div>
''', unsafe_allow_html=True)
st.markdown(''' ''')
st.markdown('''
            <div class="justified-text">
Berdasarkan evaluasi Korelasi, RMSE, dan Bias, GCM yang memiliki kombinasi nilai korelasi tinggi, dan error yang rendah adalah model
            MIROC6, lalu diikuti oleh GCM MPI. Kalau dicermati lebih dalam lagi, GCM dengan resolusi spasial rendah seperti (MIROC6, MPI, dan EC-EARTH),
            memiliki nilai evaluasi yang bagus dibandingkan dengan GCM dengan resolusi spasial rendah (ACCESS, ACCESS-CM2, CANESM5, dan CMCC).
            Hal tersebut dipengaruhi oleh bagaimana proses model CNN atau CNN-LSTM yang mempertimbangkan nilai error utk menentukan "loss"
            dalam setiap proses training model, dimana semakin tinggi resolusi GCM maka akumulasi nilai errornya pun akan semakin tinggi.
            Hal tersebut tentu saja akan mempengaruhi nilai "loss" yang bisa menyebabkan adanya Overfitting Model. 
            </div>
''', unsafe_allow_html=True)

# judul section 1
st.header('2. Saran')

# narasi pendahuluan
st.warning('''
            Perlu menguji model CNN dan CNN-LSTM yang sudah dibangun sebelumnya dengan GCM CMIP6 lainnya baik yang memiliki
           resolusi spasial rendah dan resolusi spasial tinggi. Kemudian perlu melakukan rekonstruksi arsitektur model
           Deep Learning lainnya seperti ConvLSTM, Conv2D atau Conv3D untuk mencari model terbaik yang bisa mengahasilkan
           nilai evaluasi model yang baik serta meningkatkan resolusi spasial GCM untuk semua inputan GCM yang ada.
''')

st.header('Daftar Pustaka')
col1, col2 = st.columns([1, 20], gap='small')
with col1:
    st.markdown('''**1.**''')
with col2:
    st.markdown('''
                C. Liu, Y. Zhang, and Z. Wang, "Projection of Sea Level Change in the South China Sea Based on CMIP6 Models,"
                Atmosphere, vol. 14, no. 9, p. 1343, 2023.
                ''')
col1, col2 = st.columns([1, 20], gap='small')
with col1:
    st.markdown('''**2.**''')
with col2:
    st.markdown('''
                S. K. Mishra, "Predicting Sea Level Rise Using Artificial Intelligence: A Review,"
                Journal of Marine Science and Engineering, vol. 11, no. 5, p. 789, 2023.''')
col1, col2 = st.columns([1, 20], gap='small')
with col1:
    st.markdown('''**3.**''')
with col2:
    st.markdown('''
                Wang, G., Wang, X., Wu, X., Liu, K., Qi, Y., Sun, C., & Fu, H. (2022).
                A hybrid multivariate deep learning network for multistep ahead sea level anomaly forecasting.
                Journal of Atmospheric and Oceanic Technology, 39(3).
                ''')