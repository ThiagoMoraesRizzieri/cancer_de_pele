# -*- coding: utf-8 -*-
"""
Classificador de Pintas de Pele - CNN com Streamlit
Autor: [Seu Nome]
Data: Novembro 2025
"""

import streamlit as st
import numpy as np
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import gdown
import os

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
            # ===== IMPORTANTE: SUBSTITUA O ID ABAIXO! =====
            # Para obter o ID do Google Drive:
            # 1. Faça upload do modelo para o Google Drive
            # 2. Botão direito no arquivo → Compartilhar
            # 3. Configurar: "Qualquer pessoa com o link" (Leitor)
            # 4. Copiar o link: https://drive.google.com/file/d/1ABC123XYZ/view
            # 5. O ID é: 1ABC123XYZ

            file_id = '1Hg2qY7VYH8r-LkxAbho9UMCVEFJbV4gn'
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

# ===== FUNÇÃO DE CLASSIFICAÇÃO =====
def classificar_pinta(img, modelo, threshold=0.6):
    """
    Classifica uma imagem de pinta

    Args:
        img: imagem PIL
        modelo: modelo Keras carregado
        threshold: confiança mínima para aceitar a classificação (padrão: 60%)

    Returns:
        classe_predita: índice da classe predita
        confianca: confiança da predição (0-100%)
        predictions: array com todas as probabilidades
        abaixo_threshold: boolean indicando se está abaixo do threshold
    """
    # Lista de classes (AJUSTE SE NECESSÁRIO!)
    classes = [
        'Melanoma', 
        'Nevo Melanocítico', 
        'Carcinoma Basocelular',
        'Queratose Actínica', 
        'Lesão Benigna', 
        'Dermatofibroma',
        'Lesão Vascular'
    ]

    # Converter para RGB (remove canal alpha se existir)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Processar imagem (AJUSTE O TAMANHO SE NECESSÁRIO!)
    img_resized = img.resize((100, 75))  # Tamanho usado no treino
    img_array = np.array(img_resized) / 255.0  # Normalização
    img_array = np.expand_dims(img_array, axis=0)  # Adicionar dimensão do batch

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
        help="Se a confiança for menor que este valor, a classificação será considerada incerta. "
             "Isso ajuda a identificar imagens inadequadas (ex: não é uma pinta, baixa qualidade, etc.)"
    )

    st.sidebar.markdown("---")

    # Informações de uso
    st.sidebar.info(
        "**Como usar:**\n\n"
        "1. Faça upload de uma foto da pinta\n"
        "2. Aguarde a análise automática\n"
        "3. Veja os resultados e probabilidades\n\n"
        "**Dica:** Use imagens claras, bem focadas e com boa iluminação para melhores resultados."
    )

    st.sidebar.markdown("---")

    # Sobre o modelo
    with st.sidebar.expander("ℹ️ Sobre o Modelo"):
        st.write(
            "Este classificador utiliza uma Rede Neural Convolucional (CNN) "
            "treinada para identificar 7 tipos diferentes de lesões de pele."
        )
        st.write("**Arquitetura:** CNN")
        st.write("**Input:** Imagens 100x75 pixels RGB")
        st.write("**Output:** 7 classes")

    # ===== UPLOAD DE IMAGEM =====
    st.header("📸 Upload da Imagem")

    uploaded_file = st.file_uploader(
        "Escolha uma imagem da pinta",
        type=['png', 'jpg', 'jpeg'],
        help="Formatos aceitos: PNG, JPG, JPEG"
    )

    # ===== PROCESSAMENTO E EXIBIÇÃO DOS RESULTADOS =====
    if uploaded_file is not None:

        # Carregar e exibir imagem
        img = Image.open(uploaded_file)

        # Layout em duas colunas
        col1, col2 = st.columns(2)

        # ===== COLUNA 1: IMAGEM =====
        with col1:
            st.subheader("📷 Imagem Analisada")
            st.image(img, use_column_width=True)

            # Informações da imagem
            with st.expander("📊 Informações da Imagem"):
                st.write(f"**Tamanho original:** {img.size[0]} x {img.size[1]} pixels")
                st.write(f"**Formato:** {img.format}")
                st.write(f"**Modo:** {img.mode}")

        # ===== COLUNA 2: RESULTADOS =====
        with col2:
            st.subheader("📊 Resultados da Análise")

            # Classificar
            with st.spinner('🔍 Analisando imagem...'):
                classe_predita, confianca, predictions, abaixo_threshold = classificar_pinta(
                    img, modelo, threshold/100
                )

            # Lista de classes (mesma ordem da função classificar_pinta)
            classes = [
                'Melanoma', 
                'Nevo Melanocítico', 
                'Carcinoma Basocelular',
                'Queratose Actínica', 
                'Lesão Benigna', 
                'Dermatofibroma',
                'Lesão Vascular'
            ]

            # ===== EXIBIR RESULTADO PRINCIPAL =====
            if abaixo_threshold:
                # Classificação incerta (abaixo do threshold)
                st.error(
                    f"⚠️ **Classificação Incerta**\n\n"
                    f"A confiança ({confianca:.1f}%) está abaixo do limite configurado ({threshold}%).\n\n"
                    f"**Possível classificação:** {classes[classe_predita]}\n\n"
                    f"**Recomendação:** Esta imagem pode não ser adequada para classificação. "
                    f"Possíveis causas:\n"
                    f"- Imagem de baixa qualidade ou desfocada\n"
                    f"- Ângulo inadequado ou iluminação ruim\n"
                    f"- Não é uma lesão de pele\n"
                    f"- Tipo de lesão diferente das classes conhecidas\n\n"
                    f"**Por favor, consulte um médico especialista para avaliação adequada.**"
                )
            else:
                # Classificação confiável
                st.success(
                    f"📌 **Classificação:** {classes[classe_predita]}\n\n"
                    f"💯 **Confiança:** {confianca:.2f}%"
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

        # Cores: vermelho para classe predita, azul para as outras
        colors = ['#ff4444' if classes[i] == classes[classe_predita] else '#66b3ff' 
                  for i in range(len(classes))]

        # Reordenar cores para corresponder ao DataFrame ordenado
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

        # Descrições das classes (pode personalizar)
        descricoes = {
            'Melanoma': '⚠️ Tipo mais grave de câncer de pele. Requer atenção médica imediata.',
            'Nevo Melanocítico': 'Pinta comum, geralmente benigna. Monitore mudanças.',
            'Carcinoma Basocelular': 'Tipo mais comum de câncer de pele. Crescimento lento.',
            'Queratose Actínica': 'Lesão pré-cancerosa causada por exposição solar.',
            'Lesão Benigna': 'Lesão não cancerosa, mas monitore mudanças.',
            'Dermatofibroma': 'Tumor benigno comum na pele.',
            'Lesão Vascular': 'Lesão relacionada a vasos sanguíneos.'
        }

        # Exibir descrição da classe predita
        with st.expander(f"ℹ️ Sobre: {classes[classe_predita]}"):
            st.write(descricoes.get(classes[classe_predita], "Informação não disponível."))
            st.warning("**Lembre-se:** Apenas um médico pode fornecer diagnóstico definitivo!")

    else:
        # Mensagem quando nenhuma imagem foi enviada
        st.info("👆 Faça upload de uma imagem para começar a análise")

        # Exemplos de imagens adequadas
        with st.expander("💡 Dicas para Melhores Resultados"):
            st.write(
                "**✅ Imagens adequadas:**\n"
                "- Foto clara e bem focada\n"
                "- Boa iluminação (natural é melhor)\n"
                "- Pinta centralizada na imagem\n"
                "- Fundo simples\n"
                "- Imagem em alta resolução\n\n"
                "**❌ Evite:**\n"
                "- Imagens desfocadas ou tremidas\n"
                "- Iluminação muito fraca ou muito forte\n"
                "- Fotos de longe (pinta muito pequena)\n"
                "- Imagens editadas ou com filtros\n"
                "- Fotos de telas de computador"
            )

    # ===== RODAPÉ =====
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center'>"
        "<p><strong>Desenvolvido com ❤️ usando Streamlit e TensorFlow</strong></p>"
        "<p style='font-size: 0.8em;'>Este é um projeto educacional. "
        "Não substitui diagnóstico médico profissional.</p>"
        "</div>",
        unsafe_allow_html=True
    )

# ===== EXECUTAR APLICAÇÃO =====
if __name__ == "__main__":
    main()
