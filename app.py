# -*- coding: utf-8 -*-
"""
Classificador de Pintas de Pele - CNN com Streamlit
Versão com Câmera Traseira Nativa para Celular
Membros do grupo: Laís, Giovana, Thiago, Uilma, Viviane
Data: Novembro 2025
"""

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import gdown
import os
import base64
from io import BytesIO

# ===== CONFIGURAÇÃO DA PÁGINA =====
st.set_page_config(
    page_title="Classificador de Pintas",
    page_icon="🔬",
    layout="wide"
)

# ===== CARREGAR MODELO =====
@st.cache_resource
def carregar_modelo():
    """
    Carrega o modelo do Google Drive (apenas na primeira vez)
    O cache garante que o modelo seja carregado apenas uma vez
    """
    modelo_path = 'meu_modelo.keras'

    # Se o modelo não existe localmente, baixar do Google Drive
    if not os.path.exists(modelo_path):
        with st.spinner('📥 Baixando modelo do Google Drive... (pode demorar alguns minutos)'):
            file_id = '1txLANRcl_00BcFWdvAg90byaHwKhxIYR'  # ← SEU ID AQUI
            url = f'https://drive.google.com/uc?id={file_id}'

            try:
                gdown.download(url, modelo_path, quiet=False)
                st.success('✅ Modelo baixado com sucesso!')
            except Exception as e:
                st.error(f'❌ Erro ao baixar modelo: {e}')
                st.info('Verifique se o ID do arquivo está correto e se as permissões estão configuradas como "Qualquer pessoa com o link"')
                st.stop()

    # Carregar o modelo
    try:
        modelo = keras.models.load_model(modelo_path)
        return modelo
    except Exception as e:
        st.error(f'❌ Erro ao carregar modelo: {e}')
        st.stop()

# ===== COMPONENTE DE CÂMERA TRASEIRA =====
def camera_traseira():
    """
    Componente HTML customizado que força câmera traseira no celular
    Permite alternar entre câmera frontal e traseira
    """
    html_code = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {
                margin: 0;
                padding: 10px;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                background: #f0f2f6;
            }
            #videoContainer {
                position: relative;
                width: 100%;
                max-width: 640px;
                margin: 0 auto;
                background: #000;
                border-radius: 12px;
                overflow: hidden;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            }
            #video {
                width: 100%;
                display: block;
            }
            #canvas {
                display: none;
            }
            .button-container {
                text-align: center;
                margin-top: 20px;
            }
            button {
                background: linear-gradient(135deg, #FF4B4B 0%, #FF6B6B 100%);
                color: white;
                border: none;
                padding: 15px 30px;
                font-size: 16px;
                font-weight: 600;
                border-radius: 8px;
                cursor: pointer;
                margin: 5px;
                transition: all 0.3s ease;
                box-shadow: 0 4px 8px rgba(255, 75, 75, 0.3);
            }
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 12px rgba(255, 75, 75, 0.4);
            }
            button:active {
                transform: translateY(0);
            }
            button:disabled {
                background: #cccccc;
                cursor: not-allowed;
                box-shadow: none;
            }
            #status {
                text-align: center;
                margin: 15px 0;
                padding: 12px;
                background: white;
                border-radius: 8px;
                color: #333;
                font-weight: 500;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .emoji {
                font-size: 24px;
                margin-right: 8px;
            }
        </style>
    </head>
    <body>
        <div id="status">🎥 Inicializando câmera...</div>
        <div id="videoContainer">
            <video id="video" autoplay playsinline muted></video>
        </div>
        <canvas id="canvas"></canvas>
        <div class="button-container">
            <button id="captureBtn" onclick="capturarFoto()">
                <span class="emoji">📸</span>Capturar Foto
            </button>
            <button id="switchBtn" onclick="trocarCamera()">
                <span class="emoji">🔄</span>Trocar Câmera
            </button>
        </div>

        <script>
            let stream = null;
            let usarCameraTraseira = true;

            async function iniciarCamera() {
                try {
                    // Parar stream anterior se existir
                    if (stream) {
                        stream.getTracks().forEach(track => track.stop());
                    }

                    // Configurar para câmera traseira (environment) ou frontal (user)
                    const constraints = {
                        video: {
                            facingMode: usarCameraTraseira ? { exact: "environment" } : "user",
                            width: { ideal: 1920 },
                            height: { ideal: 1080 }
                        },
                        audio: false
                    };

                    stream = await navigator.mediaDevices.getUserMedia(constraints);
                    const video = document.getElementById('video');
                    video.srcObject = stream;

                    const cameraType = usarCameraTraseira ? '📱 Câmera Traseira' : '🤳 Câmera Frontal';
                    document.getElementById('status').innerHTML = 
                        `✅ <strong>${cameraType}</strong> ativada`;

                } catch (err) {
                    console.error('Erro ao acessar câmera:', err);
                    
                    // Fallback para câmera frontal se traseira não funcionar
                    if (usarCameraTraseira && err.name === 'OverconstrainedError') {
                        document.getElementById('status').innerHTML = 
                            '⚠️ Câmera traseira não disponível. Tentando frontal...';
                        usarCameraTraseira = false;
                        setTimeout(iniciarCamera, 1000);
                    } else {
                        document.getElementById('status').innerHTML = 
                            '❌ <strong>Erro:</strong> ' + err.message;
                    }
                }
            }

            function trocarCamera() {
                usarCameraTraseira = !usarCameraTraseira;
                document.getElementById('status').innerHTML = '🔄 Trocando câmera...';
                iniciarCamera();
            }

            function capturarFoto() {
                const video = document.getElementById('video');
                const canvas = document.getElementById('canvas');
                const context = canvas.getContext('2d');

                // Configurar tamanho do canvas igual ao vídeo
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;

                // Desenhar frame atual do vídeo no canvas
                context.drawImage(video, 0, 0, canvas.width, canvas.height);

                // Converter para base64 JPEG (qualidade 95%)
                const imageData = canvas.toDataURL('image/jpeg', 0.95);

                // Enviar para Streamlit
                window.parent.postMessage({
                    type: 'streamlit:setComponentValue',
                    value: imageData
                }, '*');

                // Feedback visual
                document.getElementById('status').innerHTML = 
                    '✅ <strong>Foto capturada!</strong> Aguarde o processamento...';
                
                // Efeito de flash (opcional)
                video.style.opacity = '0.3';
                setTimeout(() => { video.style.opacity = '1'; }, 200);
            }

            // Iniciar câmera quando página carregar
            window.addEventListener('load', () => {
                iniciarCamera();
            });

            // Limpar stream quando sair da página
            window.addEventListener('beforeunload', () => {
                if (stream) {
                    stream.getTracks().forEach(track => track.stop());
                }
            });
        </script>
    </body>
    </html>
    """
    
    # Renderizar componente HTML
    image_data = components.html(html_code, height=650, scrolling=False)
    
    return image_data

# ===== FUNÇÃO DE CLASSIFICAÇÃO =====
def classificar_pinta(img, modelo, threshold=0.6):
    """
    Classifica uma imagem de pinta
    """
    # Lista de classes (HAM10000 dataset)
    classes = [
        'Queratose Actínica',      # 0 - akiec (pré-câncer)
        'Carcinoma Basocelular',   # 1 - bcc (câncer)
        'Lesão Benigna',           # 2 - bkl (benigna)
        'Dermatofibroma',          # 3 - df (benigna)
        'Melanoma',                # 4 - mel (câncer agressivo)
        'Nevo Melanocítico',       # 5 - nv (pinta benigna)
        'Lesão Vascular'           # 6 - vasc (benigna)
    ]

    # Converter para RGB (remove canal alpha se existir)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Processar imagem (75x100 conforme seu modelo)
    img_resized = img.resize((100, 75))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Fazer predição
    predictions = modelo.predict(img_array, verbose=0)
    classe_predita = np.argmax(predictions[0])
    confianca = predictions[0][classe_predita] * 100

    # Verificar se está abaixo do threshold
    abaixo_threshold = confianca < (threshold * 100)

    return classe_predita, confianca, predictions[0], abaixo_threshold

# ===== INTERFACE PRINCIPAL =====
def main():
    """Função principal que define toda a interface do Streamlit"""

    # ===== TÍTULO E CABEÇALHO =====
    st.title("🔬 Classificador de Pintas de Pele")
    st.markdown("### Análise automatizada usando Deep Learning (CNN)")
    st.markdown("---")

    # ===== AVISO IMPORTANTE =====
    st.warning(
        "⚠️ **AVISO IMPORTANTE**: Este resultado NÃO substitui consulta médica! "
        "Sempre procure um dermatologista qualificado para diagnóstico e tratamento adequados."
    )

    # ===== CARREGAR MODELO =====
    modelo = carregar_modelo()

    # ===== SIDEBAR COM CONFIGURAÇÕES =====
    st.sidebar.header("⚙️ Configurações")

    # Slider para threshold de confiança
    threshold = st.sidebar.slider(
        "Confiança mínima para classificação (%)",
        min_value=0,
        max_value=100,
        value=60,
        step=5,
        help="Se a confiança for menor que este valor, a classificação será considerada incerta."
    )

    st.sidebar.markdown("---")

    # Informações de uso
    st.sidebar.info(
        "**Como usar:**\n\n"
        "1. Escolha entre tirar foto ou carregar arquivo\n"
        "2. **Câmera Traseira:** Use o botão 🔄 para trocar\n"
        "3. Aguarde a análise automática\n"
        "4. Veja os resultados e probabilidades\n\n"
        "**Dica:** Use imagens claras e bem focadas."
    )

    st.sidebar.markdown("---")

    # Sobre o modelo
    with st.sidebar.expander("ℹ️ Sobre o Modelo"):
        st.write(
            "Este classificador utiliza uma Rede Neural Convolucional (CNN) "
            "treinada no dataset HAM10000 para identificar 7 tipos de lesões de pele."
        )
        st.write("**Input:** Imagens 100x75 pixels RGB")
        st.write("**Output:** 7 classes de lesões")
        st.write("**Dataset:** HAM10000 (Harvard Dataverse)")

    # ===== UPLOAD DE IMAGEM =====
    st.header("📸 Captura/Upload da Imagem")

    # Abas para escolher entre câmera e arquivo
    tab1, tab2 = st.tabs(["📷 Câmera do Celular", "📁 Carregar Arquivo"])

    img = None

    # ===== TAB 1: CÂMERA TRASEIRA =====
    with tab1:
        st.write("**📱 Tire uma foto com a câmera do celular**")
        st.info(
            "💡 **Dica:** A câmera traseira inicia automaticamente. "
            "Use o botão 🔄 **Trocar Câmera** se quiser alternar."
        )
        
        # Usar componente customizado de câmera
        image_data = camera_traseira()
        
        if image_data:
            try:
                # Remover prefixo base64 se existir
                if isinstance(image_data, str) and 'base64,' in image_data:
                    image_data = image_data.split('base64,')[1]
                
                # Decodificar base64
                image_bytes = base64.b64decode(image_data)
                img = Image.open(BytesIO(image_bytes))
                
                st.success("✅ Foto capturada com sucesso!")
                
            except Exception as e:
                st.error(f"❌ Erro ao processar imagem: {e}")
                img = None

    # ===== TAB 2: ARQUIVO =====
    with tab2:
        st.write("**Selecione um arquivo de imagem do seu dispositivo**")

        uploaded_file = st.file_uploader(
            "Escolha uma imagem da pinta",
            type=['png', 'jpg', 'jpeg'],
            help="Formatos aceitos: PNG, JPG, JPEG"
        )

        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.success("✅ Imagem carregada com sucesso!")

    # ===== PROCESSAMENTO E EXIBIÇÃO DOS RESULTADOS =====
    if img is not None:

        # Layout em duas colunas
        col1, col2 = st.columns(2)

        # ===== COLUNA 1: IMAGEM =====
        with col1:
            st.subheader("📷 Imagem Analisada")
            st.image(img, use_column_width=True)

            # Informações da imagem
            with st.expander("📊 Informações da Imagem"):
                st.write(f"**Tamanho original:** {img.size[0]} x {img.size[1]} pixels")
                st.write(f"**Formato:** {img.format if img.format else 'N/A'}")
                st.write(f"**Modo:** {img.mode}")

        # ===== COLUNA 2: RESULTADOS =====
        with col2:
            st.subheader("📊 Resultados da Análise")

            # Classificar
            with st.spinner('🔍 Analisando imagem...'):
                classe_predita, confianca, predictions, abaixo_threshold = classificar_pinta(
                    img, modelo, threshold/100
                )

            # Lista de classes
            classes = [
                'Queratose Actínica',      # 0
                'Carcinoma Basocelular',   # 1
                'Lesão Benigna',           # 2
                'Dermatofibroma',          # 3
                'Melanoma',                # 4
                'Nevo Melanocítico',       # 5
                'Lesão Vascular'           # 6
            ]

            # ===== EXIBIR RESULTADO PRINCIPAL =====
            if abaixo_threshold:
                st.error(
                    f"⚠️ **Classificação Incerta**\n\n"
                    f"A confiança ({confianca:.1f}%) está abaixo do limite configurado ({threshold}%).\n\n"
                    f"**Possível classificação:** {classes[classe_predita]}\n\n"
                    f"**Recomendação:** Tire outra foto com melhor iluminação ou consulte um médico."
                )
            else:
                # Determinar gravidade
                if classe_predita in [0, 1, 4]:  # Pré-câncer ou câncer
                    st.error(
                        f"🚨 **ATENÇÃO: Lesão potencialmente maligna detectada!**\n\n"
                        f"**Classificação:** {classes[classe_predita]}\n\n"
                        f"💯 **Confiança:** {confianca:.2f}%\n\n"
                        f"⚠️ **PROCURE UM DERMATOLOGISTA IMEDIATAMENTE!**"
                    )
                else:
                    st.success(
                        f"✅ **Classificação:** {classes[classe_predita]}\n\n"
                        f"💯 **Confiança:** {confianca:.2f}%\n\n"
                        f"**Nota:** Mesmo sendo benigna, consulte um médico para confirmação."
                    )

            # Métrica destacada
            st.metric(
                label="Classe Identificada",
                value=classes[classe_predita],
                delta=f"{confianca:.1f}% de confiança"
            )

        # ===== GRÁFICO DE PROBABILIDADES =====
        st.markdown("---")
        st.subheader("📈 Probabilidades por Classe")

        # Criar DataFrame para visualização
        df_probs = pd.DataFrame({
            'Classe': classes,
            'Probabilidade (%)': predictions * 100
        }).sort_values('Probabilidade (%)', ascending=True)

        # Criar gráfico de barras horizontais
        fig, ax = plt.subplots(figsize=(10, 6))

        # Cores: vermelho para classe predita, azul para outras
        colors_sorted = ['#ff4444' if row['Classe'] == classes[classe_predita] else '#66b3ff' 
                        for _, row in df_probs.iterrows()]

        bars = ax.barh(df_probs['Classe'], df_probs['Probabilidade (%)'], color=colors_sorted)
        ax.set_xlabel('Probabilidade (%)', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 100)
        ax.set_title('Distribuição de Probabilidades', fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3, linestyle='--')

        # Adicionar valores nas barras
        for i, (idx, row) in enumerate(df_probs.iterrows()):
            ax.text(row['Probabilidade (%)'] + 1, i, f"{row['Probabilidade (%)']:.1f}%",
                    va='center', fontsize=10, fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig)

        # ===== TABELA DETALHADA =====
        st.markdown("---")
        st.subheader("📋 Detalhamento das Probabilidades")

        # Criar tabela formatada
        df_display = pd.DataFrame({
            'Classe': classes,
            'Probabilidade': [f"{p*100:.2f}%" for p in predictions],
            'Confiança': ['█' * int(p*50) + '░' * (50 - int(p*50)) for p in predictions]
        })

        # Ordenar por probabilidade (maior para menor)
        df_display['Prob_Valor'] = predictions * 100
        df_display = df_display.sort_values('Prob_Valor', ascending=False)
        df_display = df_display.drop('Prob_Valor', axis=1)

        # Exibir tabela
        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True
        )

        # ===== INFORMAÇÕES ADICIONAIS =====
        st.markdown("---")
        st.subheader("📚 Informações sobre a Classificação")

        # Descrições das classes (baseado em HAM10000)
        descricoes = {
            'Queratose Actínica': '⚠️ Lesão pré-cancerosa causada por exposição solar. Pode evoluir para câncer. **Consulte um dermatologista!**',
            'Carcinoma Basocelular': '🔴 Tipo mais comum de câncer de pele. Crescimento lento mas requer tratamento. **Consulte um dermatologista imediatamente!**',
            'Lesão Benigna': '✅ Lesão tipo queratose benigna. Geralmente inofensiva, mas monitore mudanças.',
            'Dermatofibroma': '✅ Tumor benigno comum na pele. Nódulo fibroso inofensivo.',
            'Melanoma': '🚨 **CÂNCER DE PELE MAIS PERIGOSO!** Tipo mais agressivo de câncer de pele. **PROCURE UM MÉDICO URGENTEMENTE!**',
            'Nevo Melanocítico': '✅ Pinta comum (nevo). Geralmente benigna, mas monitore mudanças de tamanho, cor ou forma.',
            'Lesão Vascular': '✅ Lesão relacionada a vasos sanguíneos (angioma, hemangioma). Geralmente benigna.'
        }

        # Exibir descrição da classe predita
        with st.expander(f"ℹ️ Sobre: {classes[classe_predita]}"):
            st.write(descricoes.get(classes[classe_predita], "Informação não disponível."))
            st.warning("**Lembre-se:** Apenas um dermatologista pode fornecer diagnóstico definitivo!")

        # Regra ABCDE para melanoma
        with st.expander("📖 Regra ABCDE para Identificação de Melanoma"):
            st.markdown(
                "**A** - **Assimetria:** Uma metade da pinta diferente da outra\n\n"
                "**B** - **Bordas irregulares:** Bordas recortadas, chanfradas ou mal definidas\n\n"
                "**C** - **Cor variada:** Diferentes tons de marrom, preto, vermelho, branco ou azul\n\n"
                "**D** - **Diâmetro:** Maior que 6mm (tamanho de uma borracha de lápis)\n\n"
                "**E** - **Evolução:** Mudanças em tamanho, forma, cor ou sintomas (coceira, sangramento)\n\n"
                "⚠️ **Se notar qualquer um desses sinais, consulte um dermatologista!**"
            )

    else:
        # Mensagem quando nenhuma imagem foi enviada
        st.info("👆 Faça upload de uma imagem ou tire uma foto para começar a análise")

        # Exemplos de imagens adequadas
        with st.expander("💡 Dicas para Melhores Resultados"):
            st.write(
                "**✅ Imagens adequadas:**\n"
                "- Foto clara e bem focada\n"
                "- Boa iluminação natural (evite flash direto)\n"
                "- Pinta centralizada na imagem\n"
                "- Fundo simples e neutro\n"
                "- Câmera a ~15-20cm da pinta\n"
                "- Imagem sem sombras ou reflexos\n\n"
                "**❌ Evite:**\n"
                "- Imagens desfocadas ou tremidas\n"
                "- Iluminação muito fraca ou muito forte\n"
                "- Fotos de muito longe (pinta muito pequena)\n"
                "- Fotos de muito perto (desfocadas)\n"
                "- Imagens editadas ou com filtros\n"
                "- Sombras sobre a lesão"
            )

    # ===== RODAPÉ =====
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center'>"
        "<p><strong>Desenvolvido com ❤️ usando Streamlit e TensorFlow</strong></p>"
        "<p><strong>Dataset:</strong> HAM10000 (Harvard Dataverse)</p>"
        "<p style='font-size: 0.8em;'>Este é um projeto educacional. "
        "Não substitui diagnóstico médico profissional.</p>"
        "<p style='font-size: 0.7em; color: #666;'>Laís | Giovana | Thiago | Uilma | Viviane - Novembro 2025</p>"
        "</div>",
        unsafe_allow_html=True
    )

# ===== EXECUTAR APLICAÇÃO =====
if __name__ == "__main__":
    main()
