# import library
import streamlit as st

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
st.title('Conclusion')

st.divider()

# # judul section
# st.header('Kesimpulan')

# narasi utama
st.markdown('''
            <div class="justified-text">
Hasil evaluasi model CNN dan CNN-LSTM menunjukkan bahwa kedua model secara umum dapat meningkatkan Korelasi dan juga
mengurangi nilai error masing-masing GCM. Namun, model CNN terlihat cukup unggul dibandingkan model CNN-LSTM
pada hasil evaluasi metrik masing-masing GCM. Hal ini dapat terjadi karena pola temporal pada GCM tidak terlalu signifikan
dalam proses pemodelan, padahal layer dan convolution encoder pada CNN memegang peranan cukup penting
dalam membangun proses pemodelan, terutama dalam meningkatkan resolusi spasial GCM.
            </div>
''', unsafe_allow_html=True)
st.markdown(''' ''')
st.markdown('''
            <div class="justified-text">
Berdasarkan hasil evaluasi Korelasi, RMSE, dan Bias, GCM yang memiliki kombinasi nilai korelasi tinggi dan error rendah adalah model MIROC6,
diikuti oleh GCM MPI. Jika diamati lebih dalam, GCM dengan resolusi spasial rendah seperti (MIROC6, MPI, dan EC-EARTH),
memiliki nilai evaluasi yang baik dibandingkan dengan GCM dengan resolusi spasial yang lebih baik (ACCESS, CANESM5, dan CMCC).
Hal ini dipengaruhi oleh bagaimana proses model CNN atau CNN-LSTM mempertimbangkan nilai error untuk menentukan “loss” pada setiap proses pelatihan model,
dimana semakin tinggi resolusi GCM, maka semakin tinggi pula nilai akumulasi error. Hal ini tentu saja akan mempengaruhi nilai “loss” yang dapat menyebabkan terjadinya Overfitting Model.
            </div>
''', unsafe_allow_html=True)

# judul section 1
st.header('Rekomendasi')

# narasi pendahuluan
st.warning('''
Perlu dilakukan pengujian model CNN dan CNN-LSTM yang telah dibangun sebelumnya dengan GCM CMIP6 lainnya.
Kemudian perlu dilakukan rekonstruksi arsitektur
model Deep Learning lainnya seperti ConvLSTM, Conv2D atau Conv3D untuk menemukan model terbaik yang dapat menghasilkan
nilai evaluasi model yang baik dan meningkatkan resolusi spasial GCM untuk semua input GCM yang ada.
''')

st.subheader('Kontak')
st.success('''
            **email:** adityo.wicaksono@bmkg.go.id
''')

st.header('Referensi')
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
