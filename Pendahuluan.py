# import library
import streamlit as st

# Setting layout halaman
st.set_page_config(
        page_title="SLR-Indonesia",
        page_icon="🏠",
        layout="centered",
        initial_sidebar_state="expanded"
    )

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

######################################################
# Judul utama halaman
st.title('Sea Level Analysis and Projection in Indonesia')

# atribut tambahan
st.markdown('''
**Adityo Wicaksono**

*Indonesian Meteorology, Climatology and Geophysics Agency*

''')

st.divider()

# judul section
st.header('Overview')

# narasi utama (abstract)
st.info('''
Global climate change has caused significant sea level rise (SLR), especially in coastal areas of archipelagic countries such as Indonesia.
This study aims to analyze historical trends and project SLE in the Indonesian region using a Deep Learning-based artificial intelligence approach.
The Deep Learning model (CNN-LSTM) is used to capture the spatial and temporal dynamics of Sea Surface Height (SSH) data obtained from satellite observations (CMEMS) and climate model projections (GCM CMIP6) in the SSP245 and SSP585 scenarios.

SSH data is processed through a series of stages starting from slicing, normalization, to the formation of time sequences for input into the model architecture.
The model is trained using historical data (1995-2014) and evaluated to project future conditions (2021-2100).
The projection results show a consistent trend of increasing TML across all scenarios in most GCMs, with significant spatial variations between regions.

These findings are expected to contribute to the understanding of sea level dynamics in Indonesia and support adaptation planning and mitigation of climate change risks in coastal areas.
''')

# atribut tambahan
st.markdown('**Keywords:** *Sea Level Rise*, *Deep Learning*, *Indonesia*, *Climate Projection*')

# judul section 1
st.header('Background')

# narasi pendahuluan
st.markdown('''
            <div class="justified-text">
Global climate change has a significant impact on sea level rise in various regions of the world, including Indonesia, which is an archipelagic country.
The increase in global temperature causes the melting of polar ice and thermal expansion of sea water which directly triggers sea level rise (SLR).
This condition poses a serious threat to coastal ecosystems, infrastructure, and communities living in coastal areas of Indonesia. 
            </div>   
''', unsafe_allow_html=True)
st.markdown(''' ''')
st.markdown('''
            <div class="justified-text">
To understand and project the long-term impacts of this phenomenon, a data-driven scientific approach is needed that is able to capture the spatial and temporal complexity of sea level change.
Therefore, this study proposes the use of a combination of observational data and climate model projections with a Deep Learning approach, specifically the CNN model which has proven effective in handling spatial data.
            </div>   
''', unsafe_allow_html=True)
st.markdown(''' ''')

# judul section 2
st.header('Data and Method')

# narasi section 2
st.markdown('''
            <div class="justified-text">
This study uses Sea Surface Height (SSH) observation data from CMEMS (Copernicus Marine) with a spatial resolution of 0.083°x0.083°
and Sea Surface Height projection data from the CMIP6 GCM climate model which is divided into 2 resolutions, namely high resolution (<1°x1°) such as ACCESS, ACCESS-CM2, CANESM55, and CMCC.
While CMIP6 GCMs with low resolution (>1°x1°) such as EC-EARTH, MPI, and MIROC6.
Data were collected from the time span of 1995 to 2014 for historical data and from 2021 to 2100
for future projections (SSP245 and SSP585).
            </div>
''', unsafe_allow_html=True)
st.markdown(''' ''')
st.markdown('''
            <div class="justified-text">
GCM and SSH data are first sliced ​​based on the Indonesian region (90°-145° E and -15°-10° S).
Next, pre-processing is carried out in the form of:
            </div>
''', unsafe_allow_html=True)
cola, colb = st.columns([1,30], gap="small")
with cola:
    st.markdown('''**a.**''')
with colb:
    st.markdown('''
            <div class="justified-text">
Equalization of size and time series</div>''', unsafe_allow_html=True)
cola, colb = st.columns([1,30], gap="small")
with cola:
    st.markdown('''**b.**''')
with colb:
    st.markdown('''
            <div class="justified-text">
Filling missing values</div>''', unsafe_allow_html=True)
cola, colb = st.columns([1,30], gap="small")
with cola:
    st.markdown('''**c.**''')
with colb:
    st.markdown('''
            <div class="justified-text">
Data normalization using Min-Max Scaling technique</div>''', unsafe_allow_html=True)
cola, colb = st.columns([1,30], gap="small")
with cola:
    st.markdown('''**d.**''')
with colb:
    st.markdown('''
            <div class="justified-text">
Generation of training and test data sequences for CNN model input</div>''', unsafe_allow_html=True)

# gambar alur
st.image("alur load.png", caption="Pre-Processing GCM and Observation Data")

st.markdown('''
            <div class="justified-text">
This study uses a Deep Learning-based approach, specifically the Convolutional Neural Network (CNN) architecture
and in addition also tries to use a combination of Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM),
the model is run in two methods, the first is single CNN, and the second is a hybrid CNN and LSTM method.
This method was chosen because of its ability to handle data that has spatial and temporal dimensions simultaneously,
such as sea surface height (SSH) data from satellites and climate models.
            </div>
''', unsafe_allow_html=True)
st.markdown(''' ''')
st.markdown('''
            <div class="justified-text">
In general, each SSH (Sea Surface Height) input will be processed by CNN to obtain its spatial representation,
then a series of these representations are fed into LSTM to learn pattern changes over time.
In the process of learning the pattern, the CNN or CNN-LSTM model will also try to increase the spatial resolution of
the GCM CMIP6 input data following the spatial resolution of its Observation data (CMEMS) or commonly called Downscaling.
This architecture allows the model to predict future SLR conditions based on past dynamics.
            </div>
''', unsafe_allow_html=True)

st.image("alur model cnn-lstm.png", caption="Model Architecture")

# judul section 3
st.header('Manfaat')

# narasi section 3
st.markdown('''**This research is expected to provide the following contributions:**''')
st.markdown(''' ''')
col1, col2 = st.columns([1,30], gap="small")
with col1:
    st.markdown('''**1.**''')
with col2:
    st.markdown('''
                <div class="justified-text">
    Provides long-term projections of sea level in Indonesia based on future climate scenarios.</div> 
    ''', unsafe_allow_html=True)
col1, col2 = st.columns([1,30], gap="small")
with col1:
    st.markdown('''**2.**''')
with col2:
    st.markdown('''
                <div class="justified-text">
    Offers a Deep Learning based predictive modeling approach for climate analysis with high accuracy.</div> 
    ''', unsafe_allow_html=True)
col1, col2 = st.columns([1,30], gap="small")
with col1:
    st.markdown('''**3.**''')
with col2:
    st.markdown('''
                <div class="justified-text">
    Becoming an important source of information for decision making in mitigating and adapting to the impacts of climate change,
    especially in coastal areas.</div> 
    ''', unsafe_allow_html=True)
col1, col2 = st.columns([1,30], gap="small")
with col1:
    st.markdown('''**4.**''')
with col2:
    st.markdown('''
                <div class="justified-text">
    Promote the use of climate data and modern computational techniques in sustainable development planning.</div> 
    ''', unsafe_allow_html=True)
