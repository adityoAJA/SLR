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
# st.header('Conclusion')

# narasi utama
st.markdown('''
            <div class="justified-text">
The evaluation results of the CNN and CNN-LSTM models show that both models can generally increase the Correlation and also
reduce the error value of each GCM. However, the CNN model looks quite superior compared to the CNN-LSTM model
in the metric evaluation results of each GCM. This could be because the temporal pattern in the GCM is not too significant
in the model process, in fact the layers and convolution encoders in CNN play quite an important role
in building the model process, especially in increasing the spatial resolution of the GCM.
            </div>
''', unsafe_allow_html=True)
st.markdown(''' ''')
st.markdown('''
            <div class="justified-text">
Based on the evaluation of Correlation, RMSE, and Bias, the GCM that has a combination of high correlation values ​​and low error is the MIROC6 model,
followed by the MPI GCM. If observed more deeply, GCMs with low spatial resolution such as (MIROC6, MPI, and EC-EARTH),
have good evaluation values ​​compared to GCMs with better spatial resolution (ACCESS, CANESM5, and CMCC).
This is influenced by how the CNN or CNN-LSTM model process considers the error value to determine the "loss" in each model training process,
where the higher the GCM resolution, the higher the accumulated error value. This will of course affect the "loss" value which can cause Overfitting Model.
            </div>
''', unsafe_allow_html=True)

# judul section 1
st.header('Recomendation')

# narasi pendahuluan
st.warning('''
            It is necessary to test the previously built CNN and CNN-LSTM models with other CMIP6 GCMs.
            Then it is necessary to reconstruct the architecture of
other Deep Learning models such as ConvLSTM, Conv2D or Conv3D to find the best model that can produce
good model evaluation values ​​and increase the spatial resolution of GCMs for all existing GCM inputs.
''')

st.header('References')
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
