# app.py
import streamlit as st
import numpy as np
from tensorflow import keras
from PIL import Image

# Configurar página
st.set_page_config(
    page_title="Classificador de Pintas",
    page_icon="🔬",
    layout="centered"
)

# Carregar modelo
@st.cache_resource
def carregar_modelo():
    return keras.models.load_model('meu_modelo.keras')

modelo = carregar_modelo()

# Classes
classes = ['Melanoma', 'Nevo Melanocítico', 'Carcinoma Basocelular', 
           'Queratose Actínica', 'Lesão Benigna', 'Dermatofibroma', 
           'Lesão Vascular']

# Interface
st.title('🔬 Classificador de Lesões de Pele')
st.markdown('---')

# Opções de input
opcao = st.radio(
    "Como você quer enviar a imagem?",
    ["📤 Upload de arquivo", "📸 Tirar foto agora"]
)

imagem = None

if opcao == "📤 Upload de arquivo":
    uploaded_file = st.file_uploader(
        "Escolha uma imagem da pinta", 
        type=['jpg', 'jpeg', 'png']
    )
    if uploaded_file:
        imagem = Image.open(uploaded_file)

elif opcao == "📸 Tirar foto agora":
    camera_photo = st.camera_input("Tire uma foto")
    if camera_photo:
        imagem = Image.open(camera_photo)

# Processar se há imagem
if imagem is not None:
    # Criar colunas
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(imagem, caption='Imagem Original', use_column_width=True)
    
    # Processar
    img_resized = imagem.resize((100, 75))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predizer
    with st.spinner('🔍 Analisando...'):
        predictions = modelo.predict(img_array, verbose=0)
    
    classe_predita = np.argmax(predictions[0])
    confianca = predictions[0][classe_predita] * 100
    
    with col2:
        st.metric(
            label="Classificação",
            value=classes[classe_predita],
            delta=f"{confianca:.1f}% de confiança"
        )
    
    # Detalhes
    st.markdown('---')
    st.subheader('📊 Probabilidades Detalhadas')
    
    for i, classe in enumerate(classes):
        prob = predictions[0][i] * 100
        st.write(f"**{classe}**")
        st.progress(prob / 100)
        st.write(f"{prob:.2f}%")
        st.write("")
    
    # Aviso
    st.markdown('---')
    st.error(
        '⚠️ **IMPORTANTE:** Este é um modelo de IA educacional e '
        'NÃO substitui avaliação médica profissional. Sempre consulte '
        'um dermatologista qualificado para diagnóstico adequado.'
    )

else:
    st.info('👆 Escolha uma opção acima para começar a análise')

# Footer
st.markdown('---')
st.caption('Desenvolvido com TensorFlow e Streamlit')
