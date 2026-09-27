import os
import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="Ouvidoria Inteligente", layout="wide", page_icon="🏢")

# ---------- CACHE E CARREGAMENTO ----------
@st.cache_resource
def carregar_modelo(nome):
    return SentenceTransformer(nome)

@st.cache_data
def carregar_dados():
    caminho = 'manifestacoes.json'
    if not os.path.exists(caminho):
        caminho = os.path.join(os.path.dirname(__file__), 'manifestacoes.json')
    try:
        with open(caminho, 'r', encoding='utf-8') as f:
            return pd.DataFrame(json.load(f))
    except FileNotFoundError:
        st.error("Arquivo manifestacoes.json não encontrado. Certifique-se de que o arquivo está no mesmo diretório.")
        return pd.DataFrame()

df = carregar_dados()

# ---------- SIDEBAR ----------
st.sidebar.title("⚙️ Configurações IA")
modelo_nome = st.sidebar.selectbox(
    "Modelo de Embedding",
    ["paraphrase-multilingual-MiniLM-L12-v2", "sentence-transformers/all-MiniLM-L6-v2"]
)
top_k = st.sidebar.slider("Top-K resultados (Busca)", 1, 10, 5)

modelo = carregar_modelo(modelo_nome)

# Pré-computar embeddings totais
if not df.empty:
    @st.cache_data
    def computar_embeddings_base(textos, mod_name):
        return modelo.encode(textos)
    
    emb_base = computar_embeddings_base(df['texto'].tolist(), modelo_nome)

# ---------- TABS ----------
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Busca Semântica", "📋 Base Completa", "🌐 Espaço Vetorial", "🧩 Chunking"])

# === TAB 1: Busca Semântica ===
with tab1:
    st.header("Triagem de Manifestações")
    st.markdown("Digite o problema para encontrar relatos similares, agrupando demandas da população.")
    
    query = st.text_input("Descreva o problema:", placeholder="Ex: iluminação ruim na praça...")
    
    if st.button("Buscar Similares", type="primary") and query:
        q_emb = modelo.encode([query])
        sims = cosine_similarity(q_emb, emb_base)[0]
        ranking = np.argsort(sims)[::-1][:top_k]
        
        for pos, idx in enumerate(ranking, 1):
            score = sims[idx]
            cor = "🟢" if score > 0.7 else "🟡" if score > 0.5 else "🔴"
            row = df.iloc[idx]
            
            with st.container(border=True):
                st.markdown(f"**{cor} #{pos} | ID: {row['id']} | Categoria: `{row['categoria_oficial']}` | Score: {score:.2f}**")
                st.write(row['texto'])

# === TAB 2: Base Completa ===
with tab2:
    st.header("Banco de Dados da Ouvidoria")
    st.dataframe(df, use_container_width=True)
    
    if st.button("Gerar Matriz de Similaridade da Base"):
        with st.spinner("Calculando..."):
            sim_matrix = cosine_similarity(emb_base)
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(sim_matrix, cmap="Blues", vmin=0, vmax=1)
            plt.colorbar(im, ax=ax)
            ax.set_title("Similaridade entre todas as manifestações")
            st.pyplot(fig)

# === TAB 3: Espaço Vetorial ===
with tab3:
    st.header("Visualização de Clusters Semânticos")
    reducao = st.radio("Técnica de Redução:", ["PCA", "t-SNE"], horizontal=True)
    
    if st.button("Gerar Gráfico"):
        with st.spinner(f"Aplicando {reducao}..."):
            if reducao == "PCA":
                coords = PCA(n_components=2).fit_transform(emb_base)
            else:
                perp = min(5, len(df) - 1)
                coords = TSNE(n_components=2, perplexity=perp, random_state=42).fit_transform(emb_base)
            
            categorias = df['categoria_oficial'].unique()
            cores = plt.cm.Set1(np.linspace(0, 1, len(categorias)))
            cat_cor = dict(zip(categorias, cores))
            
            fig, ax = plt.subplots(figsize=(12, 8))
            for cat in categorias:
                idx = df[df['categoria_oficial'] == cat].index
                ax.scatter(coords[idx, 0], coords[idx, 1], 
                           label=cat, c=[cat_cor[cat]], s=100, edgecolors='black', alpha=0.8)
            
            for i, row in df.iterrows():
                ax.annotate(row['id'], (coords[i, 0], coords[i, 1]), 
                            xytext=(5, 5), textcoords='offset points', fontsize=8)
                
            ax.set_title(f"Espaço Semântico ({reducao}) Colorido por Categoria Oficial")
            ax.legend(title="Categorias")
            st.pyplot(fig)
            
            st.info("""
            **Análise do Desenvolvedor:** Se os clusters formados (pontos próximos) possuem a mesma cor, 
            significa que as representações semânticas da IA concordam fortemente com a 
            classificação humana original da ouvidoria.
            """)

# === TAB 4: Chunking ===
with tab4:
    st.header("Tratamento de Textos Longos (Chunking)")
    texto_longo = st.text_area("Cole a denúncia longa aqui:", height=200, 
                               value=df[df['texto'].str.len() > 500]['texto'].iloc[0] if not df.empty else "")
    
    col1, col2 = st.columns(2)
    chunk_size = col1.slider("Tamanho do Chunk", 100, 500, 200, 50)
    overlap = col2.slider("Overlap", 0, 200, 50, 10)
    
    if st.button("Processar Chunks"):
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        chunks = splitter.split_text(texto_longo)
        
        st.success(f"Dividido em {len(chunks)} partes.")
        
        emb_chunks = modelo.encode(chunks)
        
        for i, c in enumerate(chunks):
            with st.expander(f"Parte {i+1} ({len(c)} chars)"):
                st.write(c)
                
        # Mostrar similaridade sequencial
        if len(chunks) > 1:
            st.write("**Similaridade entre partes consecutivas (avaliação de contexto):**")
            for i in range(len(chunks)-1):
                sim = cosine_similarity([emb_chunks[i]], [emb_chunks[i+1]])[0][0]
                st.progress(float(sim), text=f"Parte {i+1} → Parte {i+2} : Score {sim:.2f}")
