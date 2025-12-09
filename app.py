import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
import tempfile
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords

# Téléchargement des ressources NLTK
try:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')
except:
    pass

# Configuration de la page
st.set_page_config(
    page_title="Classification de Documents Arabes",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* ===== IMPORT DE POLICES FUTURISTES ===== */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;600;800&family=Rajdhani:wght@300;500;700&display=swap');
    
    /* ===== VARIABLES DE COULEUR - THÈME CYBERPUNK ===== */
    :root {
        --primary-cyan: #00f3ff;
        --primary-blue: #0066ff;
        --primary-purple: #9d00ff;
        --neon-pink: #ff00ff;
        --neon-green: #00ff9d;
        
        --bg-space: #0a0a14;
        --bg-deep: #050510;
        --bg-card: rgba(16, 16, 32, 0.7);
        --bg-glass: rgba(255, 255, 255, 0.05);
        
        --text-primary: #ffffff;
        --text-glow: #e6f7ff;
        --text-cyan: #a0f0ff;
        --text-purple: #d0a0ff;
        
        --gradient-main: linear-gradient(135deg, var(--primary-cyan) 0%, var(--primary-blue) 50%, var(--primary-purple) 100%);
        --gradient-neon: linear-gradient(90deg, var(--neon-pink), var(--primary-cyan));
        --gradient-space: radial-gradient(circle at 30% 20%, rgba(0, 102, 255, 0.15) 0%, transparent 50%),
                         radial-gradient(circle at 70% 80%, rgba(157, 0, 255, 0.15) 0%, transparent 50%);
        
        --border-glow: 0 0 15px var(--primary-cyan);
        --shadow-hologram: 0 0 30px rgba(0, 243, 255, 0.3);
        --shadow-deep: 0 20px 60px rgba(0, 0, 0, 0.6);
        
        --shape-blob: polygon(0% 0%, 100% 0%, 100% 85%, 85% 100%, 0% 100%);
        --shape-cyber: polygon(0% 0%, 90% 0%, 100% 10%, 100% 100%, 10% 100%, 0% 90%);
        --shape-hexagon: polygon(25% 0%, 75% 0%, 100% 50%, 75% 100%, 25% 100%, 0% 50%);
        --shape-wave: polygon(0% 0%, 100% 0%, 100% 80%, 80% 100%, 0% 100%);
    }
    
    /* ===== ANIMATIONS AVANCÉES ===== */
    @keyframes hologramFloat {
        0%, 100% { 
            transform: translateY(0) rotate(0deg); 
            filter: drop-shadow(0 5px 15px rgba(0, 243, 255, 0.3));
        }
        25% { 
            transform: translateY(-10px) rotate(0.5deg); 
            filter: drop-shadow(0 10px 25px rgba(157, 0, 255, 0.4));
        }
        50% { 
            transform: translateY(-5px) rotate(-0.5deg);
            filter: drop-shadow(0 15px 30px rgba(0, 102, 255, 0.5));
        }
        75% { 
            transform: translateY(-8px) rotate(0.3deg);
        }
    }
    
    @keyframes scanline {
        0% { transform: translateY(-100%); }
        100% { transform: translateY(100%); }
    }
    
    @keyframes pulseGlow {
        0%, 100% { opacity: 0.3; }
        50% { opacity: 0.8; }
    }
    
    @keyframes textShimmer {
        0% { background-position: -200% center; }
        100% { background-position: 200% center; }
    }
    
    @keyframes morphShape {
        0%, 100% { clip-path: var(--shape-blob); }
        33% { clip-path: var(--shape-cyber); }
        66% { clip-path: var(--shape-hexagon); }
    }
    
    /* ===== FOND SPATIAL AVANCÉ ===== */
    .stApp {
        background: var(--bg-space);
        background-image: 
            var(--gradient-space),
            linear-gradient(45deg, transparent 95%, rgba(0, 243, 255, 0.1) 100%),
            linear-gradient(135deg, transparent 95%, rgba(157, 0, 255, 0.1) 100%);
        font-family: 'Exo 2', sans-serif;
        color: var(--text-primary);
        min-height: 100vh;
        position: relative;
        overflow-x: hidden;
    }
    
    /* ===== SCANLINE EFFET ===== */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, 
            transparent 0%, 
            var(--primary-cyan) 50%, 
            transparent 100%);
        z-index: 1000;
        animation: scanline 8s linear infinite;
        pointer-events: none;
    }
    
    /* ===== TYPOGRAPHIE FUTURISTE ===== */
    h1, .main-header {
        font-family: 'Orbitron', monospace !important;
        font-weight: 900;
        font-size: 4.5rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        background: var(--gradient-main);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(0, 243, 255, 0.5);
        margin-bottom: 2rem;
        position: relative;
        line-height: 1.1;
        animation: hologramFloat 8s ease-in-out infinite;
    }
    
    h2, .sub-header {
        font-family: 'Rajdhani', sans-serif !important;
        font-weight: 700;
        font-size: 2.8rem;
        letter-spacing: 2px;
        color: var(--text-cyan);
        position: relative;
        padding-left: 2rem;
        margin: 3rem 0 2rem;
        border-left: 4px solid var(--primary-cyan);
        text-transform: uppercase;
        background: linear-gradient(90deg, var(--text-cyan), var(--text-primary));
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    h3 {
        font-family: 'Exo 2', sans-serif !important;
        font-weight: 600;
        font-size: 2rem;
        color: var(--text-primary);
        position: relative;
        margin: 2rem 0 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px dashed rgba(0, 243, 255, 0.3);
    }
    
    p, .stMarkdown, .stText {
        font-family: 'Exo 2', sans-serif !important;
        font-weight: 300;
        font-size: 1.2rem;
        line-height: 1.8;
        color: var(--text-glow) !important;
        letter-spacing: 0.3px;
        margin-bottom: 1.5rem;
        max-width: 800px;
    }
    
    /* ===== EN-TÊTE PRINCIPAL HOLOGRAM ===== */
    .main-header {
        text-align: center;
        padding: 4rem 3rem;
        margin: 3rem auto;
        background: rgba(16, 16, 32, 0.6);
        backdrop-filter: blur(20px);
        border: 2px solid rgba(0, 243, 255, 0.2);
        clip-path: polygon(0% 0%, 95% 0%, 100% 10%, 100% 90%, 95% 100%, 5% 100%, 0% 90%, 0% 10%);
        position: relative;
        overflow: hidden;
        max-width: 1200px;
        animation: morphShape 15s ease-in-out infinite, hologramFloat 8s ease-in-out infinite;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: var(--gradient-main);
        z-index: -1;
        filter: blur(20px);
        opacity: 0.3;
        clip-path: inherit;
    }
    
    .main-header::after {
        content: '𝐒𝐘𝐒𝐓𝐄𝐌';
        font-family: 'Orbitron', monospace;
        font-size: 1rem;
        letter-spacing: 5px;
        color: var(--primary-cyan);
        display: block;
        margin-top: 1rem;
        position: relative;
        animation: pulseGlow 2s ease-in-out infinite;
    }
            
    /* ===== HEADER FLOTTANT SUPERPOSÉ ===== */
    .header-container {
        position: relative;
        width: 100%;
        margin-bottom: 5rem;
    }

    .floating-text {
        position: absolute;
        top: -40px;
        left: 50%;
        transform: translateX(-50%);
        font-family: 'Orbitron', monospace !important;
        font-weight: 900;
        font-size: 3.2rem;
        background: var(--gradient-main);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 
            0 0 30px rgba(0, 243, 255, 0.5),
            0 0 60px rgba(0, 243, 255, 0.3);
        z-index: 100;
        text-align: center;
        white-space: nowrap;
        padding: 0.8rem 2.5rem;
        backdrop-filter: blur(10px);
        border: 2px solid rgba(0, 243, 255, 0.3);
        clip-path: polygon(0% 0%, 95% 0%, 100% 25%, 100% 75%, 95% 100%, 5% 100%, 0% 75%, 0% 25%);
        animation: floatGlow 4s ease-in-out infinite;
    }

    @keyframes floatGlow {
        0%, 100% { 
            transform: translateX(-50%) translateY(0);
            box-shadow: 0 10px 30px rgba(0, 243, 255, 0.2);
        }
        50% { 
            transform: translateX(-50%) translateY(-10px);
            box-shadow: 0 20px 50px rgba(0, 243, 255, 0.4);
        }
    }

    .floating-text::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: var(--gradient-main);
        z-index: -1;
        filter: blur(15px);
        opacity: 0.3;
        clip-path: inherit;
    }

    /* ===== VERSION ALTERNATIVE : TEXTE DÉCALÉ ===== */
    .floating-text-alt {
        position: relative;
        display: inline-block;
        margin-bottom: 3rem;
    }

    .floating-text-alt::before {
        content: '🤖 Classification de Documents Arabes';
        position: absolute;
        top: -15px;
        left: -20px;
        font-family: 'Orbitron', monospace;
        font-weight: 900;
        font-size: 3.5rem;
        background: var(--gradient-main);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        z-index: 10;
        opacity: 0.9;
    }

    .floating-text-alt::after {
        content: '🤖 Classification de Documents Arabes';
        position: absolute;
        top: 15px;
        left: 20px;
        font-family: 'Orbitron', monospace;
        font-weight: 900;
        font-size: 3.5rem;
        color: rgba(0, 243, 255, 0.2);
        z-index: 5;
    }

    /* ===== VERSION 3 : TEXTE AVEC EFFET DE PROJECTION ===== */
    .text-projection {
        position: relative;
        font-family: 'Orbitron', monospace;
        font-weight: 900;
        font-size: 3.5rem;
        text-align: center;
        margin-bottom: 4rem;
        perspective: 1000px;
    }

    .text-projection .front-layer {
        position: relative;
        background: var(--gradient-main);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        transform: translateZ(50px);
        z-index: 2;
        animation: textTilt 6s ease-in-out infinite;
    }

    .text-projection .back-layer {
        position: absolute;
        top: 10px;
        left: 0;
        width: 100%;
        background: var(--gradient-main);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        filter: blur(10px);
        opacity: 0.5;
        transform: translateZ(0);
        z-index: 1;
    }

    @keyframes textTilt {
        0%, 100% { transform: translateZ(50px) rotateX(0deg); }
        25% { transform: translateZ(50px) rotateX(5deg) rotateY(5deg); }
        75% { transform: translateZ(50px) rotateX(-5deg) rotateY(-5deg); }
    }

    /* ===== MODIFICATION DU MAIN-HEADER EXISTANT ===== */
    .main-header {
        /* Réduisez le padding-top pour compenser le texte flottant */
        padding-top: 2rem !important;
        margin-top: 4rem !important;
    }

    .main-header::before {
        /* Ajustez le contenu pour ne pas interférer avec le texte flottant */
        content: 'SYSTEM';
        font-size: 0.9rem;
        letter-spacing: 4px;
        color: var(--primary-cyan);
        display: block;
        margin-bottom: 0.5rem;
        animation: pulseGlow 2s ease-in-out infinite;
    }
    
    /* ===== LAYOUT INNOVANT - GRID ORGANIQUE ===== */
    .cyber-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 2rem;
        margin: 3rem 0;
        position: relative;
    }
    
    .cyber-grid::before {
        content: '';
        position: absolute;
        top: -20px;
        left: -20px;
        right: -20px;
        bottom: -20px;
        background: 
            linear-gradient(90deg, transparent 95%, rgba(0, 243, 255, 0.1) 100%),
            linear-gradient(180deg, transparent 95%, rgba(157, 0, 255, 0.1) 100%);
        z-index: -1;
        border-radius: 20px;
    }
    
    /* ===== CARTES MORPHING ===== */
    .metric-card {
        background: rgba(16, 16, 32, 0.8);
        backdrop-filter: blur(15px);
        padding: 2.5rem;
        border: 1px solid rgba(0, 243, 255, 0.2);
        clip-path: polygon(0% 0%, 92% 0%, 100% 8%, 100% 100%, 8% 100%, 0% 92%);
        transition: all 0.5s cubic-bezier(0.34, 1.56, 0.64, 1);
        position: relative;
        overflow: hidden;
        min-height: 180px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .metric-card:hover {
        clip-path: polygon(0% 0%, 100% 0%, 100% 20%, 92% 100%, 0% 100%);
        transform: translateY(-10px) scale(1.03);
        border-color: var(--primary-cyan);
        box-shadow: var(--shadow-hologram), 0 20px 40px rgba(0, 0, 0, 0.4);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(0, 243, 255, 0.1), 
            transparent);
        transition: 0.6s;
    }
    
    .metric-card:hover::before {
        left: 100%;
    }
    
    .metric-value {
        font-family: 'Orbitron', monospace !important;
        font-size: 3.5rem;
        font-weight: 700;
        background: var(--gradient-main);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0;
        line-height: 1;
        text-shadow: 0 0 20px rgba(0, 243, 255, 0.3);
    }
    
    .metric-label {
        font-family: 'Rajdhani', sans-serif !important;
        font-size: 0.9rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: var(--text-cyan) !important;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    /* ===== ZONES DE TEXTE ORGANIQUES ===== */
    .text-flow {
        background: rgba(16, 16, 32, 0.6);
        backdrop-filter: blur(10px);
        padding: 3rem;
        margin: 2.5rem 0;
        border: 1px solid rgba(0, 243, 255, 0.1);
        clip-path: polygon(0% 0%, 100% 0%, 100% 90%, 90% 100%, 0% 100%);
        position: relative;
        transition: all 0.4s ease;
    }
    
    .text-flow:hover {
        clip-path: polygon(0% 0%, 100% 0%, 100% 85%, 85% 100%, 0% 100%);
        border-color: rgba(0, 243, 255, 0.3);
        transform: translateY(-5px);
    }
    
    .text-flow::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: var(--gradient-main);
        clip-path: polygon(0% 0%, 100% 0%, 90% 100%, 0% 100%);
    }
    
    /* ===== TEXTE ARABE FUTURISTE ===== */
    .arabic-text {
        direction: rtl;
        text-align: justify;
        font-family: 'Exo 2', 'Segoe UI', sans-serif;
        font-size: 1.6rem;
        line-height: 2.2;
        padding: 3rem;
        margin: 2.5rem 0;
        background: rgba(16, 16, 32, 0.7);
        backdrop-filter: blur(15px);
        border: 2px solid;
        border-image: linear-gradient(45deg, var(--primary-cyan), var(--primary-purple)) 1;
        clip-path: polygon(0% 0%, 98% 0%, 100% 5%, 100% 95%, 98% 100%, 2% 100%, 0% 95%, 0% 5%);
        position: relative;
        color: var(--text-primary) !important;
    }
    
    .arabic-text::before {
        content: '𐍈';
        position: absolute;
        top: -20px;
        right: 20px;
        font-size: 2rem;
        color: var(--primary-cyan);
        font-family: 'Orbitron', monospace;
    }
    
    /* ===== BOUTONS CYBER MORPH ===== */
    .stButton > button {
        font-family: 'Rajdhani', sans-serif !important;
        font-weight: 600;
        font-size: 1.1rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        background: transparent;
        border: 2px solid var(--primary-cyan);
        color: var(--primary-cyan) !important;
        padding: 1.2rem 3rem;
        clip-path: polygon(0% 0%, 90% 0%, 100% 25%, 100% 100%, 10% 100%, 0% 75%);
        transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(5px);
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(0, 243, 255, 0.2), 
            transparent);
        transition: 0.6s;
    }
    
    .stButton > button:hover {
        background: var(--gradient-main);
        color: var(--bg-space) !important;
        clip-path: polygon(0% 0%, 100% 0%, 100% 20%, 90% 100%, 0% 100%);
        transform: translateY(-3px) scale(1.05);
        box-shadow: var(--shadow-hologram);
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button[kind="primary"] {
        background: var(--gradient-main);
        color: var(--bg-space) !important;
        border: none;
        font-weight: 700;
        animation: pulseGlow 2s ease-in-out infinite;
    }
    
    /* ===== SIDEBAR CYBER MORPH ===== */
    [data-testid="stSidebar"] {
        background: rgba(5, 5, 16, 0.9) !important;
        backdrop-filter: blur(25px);
        border-right: 2px solid;
        border-image: linear-gradient(to bottom, var(--primary-cyan), var(--primary-purple)) 1;
    }
    
    .sidebar-title {
        font-family: 'Orbitron', monospace !important;
        font-size: 2.2rem;
        font-weight: 700;
        text-align: center;
        padding: 2rem 1rem;
        margin-bottom: 3rem;
        background: linear-gradient(90deg, var(--primary-cyan), var(--text-primary));
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        position: relative;
        text-transform: uppercase;
        letter-spacing: 3px;
    }
    
    .sidebar-title::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 25%;
        width: 50%;
        height: 3px;
        background: var(--gradient-main);
        clip-path: polygon(0% 0%, 100% 0%, 90% 100%, 10% 100%);
    }
    
    /* ===== ONGLETS MORPHING ===== */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(16, 16, 32, 0.6);
        backdrop-filter: blur(10px);
        padding: 1rem;
        border: 1px solid rgba(0, 243, 255, 0.2);
        clip-path: polygon(0% 0%, 98% 0%, 100% 20%, 100% 100%, 0% 100%);
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
        font-size: 1rem;
        letter-spacing: 1px;
        background: transparent;
        border: none;
        color: var(--text-cyan) !important;
        padding: 1rem 2rem;
        margin: 0 0.5rem;
        clip-path: polygon(0% 0%, 90% 0%, 100% 50%, 90% 100%, 0% 100%);
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(0, 243, 255, 0.1);
        transform: translateY(-2px);
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: var(--gradient-main);
        color: var(--bg-space) !important;
        clip-path: polygon(0% 0%, 100% 0%, 100% 20%, 90% 100%, 0% 100%);
        box-shadow: var(--border-glow);
        font-weight: 700;
    }
    
    /* ===== INPUTS CYBER ===== */
    .stTextArea textarea,
    .stTextInput input {
        font-family: 'Exo 2', sans-serif !important;
        background: rgba(16, 16, 32, 0.7) !important;
        backdrop-filter: blur(10px);
        border: 2px solid rgba(0, 243, 255, 0.3) !important;
        color: var(--text-primary) !important;
        padding: 1.2rem 1.5rem !important;
        clip-path: polygon(0% 0%, 95% 0%, 100% 25%, 100% 100%, 5% 100%, 0% 75%);
        transition: all 0.3s ease;
        font-size: 1.1rem;
    }
    
    .stTextArea textarea:focus,
    .stTextInput input:focus {
        border-color: var(--primary-cyan) !important;
        box-shadow: 0 0 20px rgba(0, 243, 255, 0.3) !important;
        clip-path: polygon(0% 0%, 100% 0%, 100% 20%, 95% 100%, 0% 100%);
        transform: translateY(-2px);
    }
    
    /* ===== EXPANDER MORPH ===== */
    .streamlit-expanderHeader {
        font-family: 'Rajdhani', sans-serif !important;
        font-weight: 600;
        font-size: 1.2rem;
        letter-spacing: 1px;
        background: rgba(16, 16, 32, 0.7);
        backdrop-filter: blur(10px);
        border: 2px solid rgba(0, 243, 255, 0.2);
        clip-path: polygon(0% 0%, 95% 0%, 100% 25%, 100% 100%, 0% 100%);
        color: var(--text-cyan) !important;
        padding: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(0, 243, 255, 0.1);
        border-color: var(--primary-cyan);
        clip-path: polygon(0% 0%, 100% 0%, 100% 20%, 95% 100%, 0% 100%);
        transform: translateY(-2px);
    }
    
    .streamlit-expanderContent {
        background: rgba(5, 5, 16, 0.8);
        backdrop-filter: blur(15px);
        border: 2px solid rgba(0, 243, 255, 0.2);
        border-top: none;
        clip-path: polygon(0% 0%, 100% 0%, 100% 95%, 95% 100%, 0% 100%);
        padding: 2rem;
    }
    
    /* ===== ALERTES CYBER ===== */
    .stAlert {
        font-family: 'Exo 2', sans-serif !important;
        background: rgba(16, 16, 32, 0.8);
        backdrop-filter: blur(15px);
        border: 2px solid;
        clip-path: polygon(0% 0%, 98% 0%, 100% 15%, 100% 100%, 0% 100%);
        padding: 2rem;
        margin: 2rem 0;
    }
    
    .stAlert.success {
        border-image: linear-gradient(45deg, var(--neon-green), var(--primary-cyan)) 1;
    }
    
    .stAlert.error {
        border-image: linear-gradient(45deg, #ff0066, var(--neon-pink)) 1;
    }
    
    .stAlert.info {
        border-image: linear-gradient(45deg, var(--primary-cyan), var(--primary-blue)) 1;
    }
    
    /* ===== FOOTER HOLOGRAM ===== */
    .footer {
        font-family: 'Rajdhani', sans-serif !important;
        text-align: center;
        padding: 4rem 2rem;
        margin-top: 5rem;
        background: rgba(16, 16, 32, 0.6);
        backdrop-filter: blur(20px);
        border-top: 2px solid;
        border-image: linear-gradient(90deg, transparent, var(--primary-cyan), transparent) 1;
        clip-path: polygon(0% 0%, 100% 0%, 100% 90%, 90% 100%, 10% 100%, 0% 90%);
        position: relative;
    }
    
    .footer::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: var(--gradient-main);
        clip-path: polygon(0% 0%, 100% 0%, 90% 100%, 10% 100%);
    }
    
    .footer-brand {
        font-family: 'Orbitron', monospace !important;
        font-size: 2.5rem;
        font-weight: 900;
        letter-spacing: 4px;
        background: var(--gradient-main);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
        text-transform: uppercase;
    }
    
    .footer-subtitle {
        font-size: 1.1rem;
        letter-spacing: 2px;
        color: var(--text-cyan) !important;
        text-transform: uppercase;
        margin-bottom: 2rem;
    }
    
    /* ===== EFFETS DE PARTICULES ===== */
    .particles {
        position: fixed;
        pointer-events: none;
        z-index: -1;
    }
    
    .particle {
        position: absolute;
        background: var(--primary-cyan);
        border-radius: 50%;
        animation: float 20s infinite linear;
    }
    
    /* ===== ORGANISATION DES PARAGRAPHES ===== */
    .paragraph-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 2.5rem;
        margin: 3rem 0;
    }
    
    .paragraph-card {
        background: rgba(16, 16, 32, 0.5);
        backdrop-filter: blur(10px);
        padding: 2.5rem;
        border: 1px solid rgba(0, 243, 255, 0.1);
        clip-path: polygon(0% 0%, 100% 0%, 100% 85%, 85% 100%, 0% 100%);
        transition: all 0.4s ease;
    }
    
    .paragraph-card:hover {
        clip-path: polygon(0% 0%, 100% 0%, 100% 80%, 80% 100%, 0% 100%);
        transform: translateY(-8px);
        border-color: var(--primary-cyan);
    }
    
    .paragraph-number {
        font-family: 'Orbitron', monospace;
        font-size: 2rem;
        color: var(--primary-cyan);
        margin-bottom: 1rem;
        font-weight: 700;
    }
    
    /* ===== RESPONSIVE DESIGN ===== */
    @media (max-width: 1200px) {
        .main-header { font-size: 3.5rem; }
        .cyber-grid { grid-template-columns: repeat(2, 1fr); }
    }
    
    @media (max-width: 768px) {
        .main-header { 
            font-size: 2.5rem; 
            padding: 2.5rem 1.5rem;
            clip-path: polygon(0% 0%, 98% 0%, 100% 5%, 100% 95%, 98% 100%, 2% 100%, 0% 95%, 0% 5%);
        }
        
        .cyber-grid { grid-template-columns: 1fr; }
        .metric-card { padding: 2rem; }
        .metric-value { font-size: 2.8rem; }
        
        .stButton > button {
            padding: 1rem 2rem;
            font-size: 1rem;
        }
        
        .arabic-text {
            font-size: 1.4rem;
            padding: 2rem;
        }
    }
    
    @media (max-width: 480px) {
        .main-header { font-size: 2rem; }
        .sub-header { font-size: 1.8rem; }
        .metric-value { font-size: 2.2rem; }
        .paragraph-grid { grid-template-columns: 1fr; }
    }
    
    /* ===== EFFETS SPÉCIAUX ===== */
    .glitch-text {
        position: relative;
        animation: glitch 3s infinite;
    }
    
    .hologram {
        background: linear-gradient(45deg, 
            transparent 45%, 
            rgba(0, 243, 255, 0.1) 50%, 
            transparent 55%);
        background-size: 200% 200%;
        animation: hologramFloat 6s ease-in-out infinite;
    }
    
    /* ===== SCROLLBAR FUTURISTE ===== */
    ::-webkit-scrollbar {
        width: 12px;
        background: rgba(16, 16, 32, 0.5);
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0, 243, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--gradient-main);
        border-radius: 10px;
        border: 2px solid var(--bg-space);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--gradient-neon);
    }
</style>
""", unsafe_allow_html=True)
# ==================== CLASSES DE PRÉTRAITEMENT ====================
class ArabicTextNormalizer:
    
    def __init__(self):
        self.alif_variations = ['أ', 'إ', 'آ', 'ٱ', 'ا']
        self.yae_variations = ['ى', 'ئ', 'ي']
        self.tae_variations = ['ة', 'ه']
        self.arabic_punctuation = '،؛؟ـ«»'
        self.extended_punctuation = self.arabic_punctuation + '!"#$%&\'()*+,-./:;<=>@[\\]^_`{|}~'
        
    def normalize_alif(self, text):
        for variation in self.alif_variations[1:]:
            text = text.replace(variation, self.alif_variations[0])
        return text
    
    def normalize_yae(self, text):
        for variation in self.yae_variations[1:]:
            text = text.replace(variation, self.yae_variations[0])
        return text
    
    def normalize_tae(self, text):
        text = text.replace(self.tae_variations[0], self.tae_variations[1])
        return text
    
    def remove_diacritics(self, text):
        diacritics = re.compile('[\u064B-\u065F\u0670]')
        return diacritics.sub('', text)

    def remove_digits(self, text): 
        text = re.sub(r'\d+', ' ', text)
        return text
    
    def normalize_spaces(self, text):
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    def remove_punctuation(self, text):
        return re.sub(f'[{re.escape(self.extended_punctuation)}]', ' ', text)
    
    def normalize_text(self, text, 
                      normalize_chars=True,
                      remove_diacritics_flag=True,
                      remove_punct=True, 
                      remove_digits_flag=True):
        
        if normalize_chars:
            text = self.normalize_alif(text)
            text = self.normalize_yae(text)
            text = self.normalize_tae(text)
        
        if remove_diacritics_flag:
            text = self.remove_diacritics(text)
        
        if remove_punct:
            text = self.remove_punctuation(text)

        if remove_digits_flag:
            text = self.remove_digits(text)
        
        text = self.normalize_spaces(text)
        
        return text

class ArabicTokenizer:
    def __init__(self):
        try:
            self.arabic_stopwords = set(stopwords.words('arabic'))
        except:
            self.arabic_stopwords = set()
    
    def tokenize(self, text):
        tokens = text.split()
        if self.arabic_stopwords:
            tokens = [token for token in tokens if token not in self.arabic_stopwords]
        return tokens

# ==================== FONCTIONS DE PRÉTRAITEMENT ====================
arabic_text_normalizer = ArabicTextNormalizer()
arabic_tokenizer = ArabicTokenizer()

def arabic_preprocessing(text):
    """Prétraitement complet du texte arabe"""
    # Normalisation
    normalized_text = arabic_text_normalizer.normalize_text(text)
    
    # Tokenisation
    tokens = arabic_tokenizer.tokenize(normalized_text)
    
    # Reconstruire le texte
    return " ".join(tokens)

# ==================== CHARGEMENT DES MODÈLES ====================
@st.cache_resource
def load_svc_model():
    """Charge le modèle LinearSVC pré-entraîné"""
    try:
        # Chemin absolu du modèle Linear SVC
        svc_model_path = os.path.join("models", "linear_svc.pkl")
        
        # Vérifier l'existence du fichier
        if not os.path.exists(svc_model_path):
            st.error(f"❌ Fichier modèle LinearSVC introuvable: {svc_model_path}")
            st.error("Veuillez vérifier que le fichier existe à cet emplacement.")
            return None
        
        # Charger le modèle LinearSVC
        model = joblib.load(svc_model_path)
        
        # Extraire le vectorizer du modèle si disponible
        if hasattr(model, 'named_steps') and 'tfidfvectorizer' in model.named_steps:
            vectorizer = model.named_steps['tfidfvectorizer']
        elif hasattr(model, 'vectorizer'):
            vectorizer = model.vectorizer
        elif hasattr(model, '_vectorizer'):
            vectorizer = model._vectorizer
        else:
            # Chercher un vectorizer séparé
            vectorizer_path = os.path.join("vectorizer", "tfidf_vectorizer.pkl")
            if os.path.exists(vectorizer_path):
                vectorizer = joblib.load(vectorizer_path)
            else:
                st.error("❌ Vectorizer TF-IDF introuvable")
                return None
    
        return model, vectorizer
        
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle: {str(e)}")
        return None, None

# ==================== FONCTION DE CLASSIFICATION ====================
def classify_with_svc(text, model, vectorizer):
    """Classification avec le modèle LinearSVC"""
    try:
        # Prétraiter le texte
        cleaned_text = arabic_preprocessing(text)
        
        # Vectoriser le texte
        text_vectorized = vectorizer.transform([cleaned_text])
        
        # Vérifier la compatibilité des dimensions
        expected_features = None
        if hasattr(model, 'coef_'):
            expected_features = model.coef_.shape[1]
        elif hasattr(model, 'named_steps') and 'linearsvc' in model.named_steps and hasattr(model.named_steps['linearsvc'], 'coef_'):
            expected_features = model.named_steps['linearsvc'].coef_.shape[1]
        actual_features = text_vectorized.shape[1]
        if expected_features is not None and actual_features != expected_features:
            st.error(f"❌ Incompatibilité de dimensions: {actual_features} ≠ {expected_features}")
            return None, None
        
        # Prédiction
        prediction = model.predict(text_vectorized)[0]
        
        # Scores de décision (LinearSVC utilise decision_function)
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(text_vectorized)[0]
        elif hasattr(model, 'decision_function'):
            decision_scores = model.decision_function(text_vectorized)[0]

            # Convertir en probabilités avec softmax
            exp_scores = np.exp(decision_scores - np.max(decision_scores))
            probabilities = exp_scores / np.sum(exp_scores)
        else:
            # Fallback: probabilités uniformes si decision_function non disponible
            probabilities = np.ones(len(model.classes_)) / len(model.classes_)
        
        # Récupérer les noms des catégories
        if hasattr(model, 'classes_'):
            category_names = list(model.classes_)
        else:
            # Catégories par défaut (basées sur votre dataset)
            category_names = ['Culture', 'Finance', 'Medical', 'Politics', 'Religion', 'Sports', 'Tech']
        
        # Créer le dictionnaire de scores
        scores = {category_names[i]: float(probabilities[i]) for i in range(len(category_names))}
        
        return scores, category_names[prediction] if prediction < len(category_names) else "Inconnu"
    
    except Exception as e:
        st.error(f"❌ Erreur lors de la classification: {str(e)}")
        return None, None
    

# ==================== INTERFACE ====================
# Sidebar pour la navigation
with st.sidebar:
    # Logo et titre
    st.markdown('<div class="sidebar-title">SNI TASNEEF</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Navigation
    page = st.radio(
        "Navigation:",
        ["Accueil", "Test en Temps Réel"],
        label_visibility="collapsed"
    )

    # Charger les modèles
    if 'svc_model_loaded' not in st.session_state:
        with st.spinner("Chargement du modèle LinearSVC..."):
            model, vectorizer = load_svc_model()
            if model and vectorizer:
                st.session_state.svc_model = model
                st.session_state.svc_vectorizer = vectorizer
                st.session_state.svc_model_loaded = True
                #st.success("✅ Modèle chargé")
            else:
                #st.error("❌ Échec du chargement")
                st.session_state.svc_model = None
                st.session_state.svc_vectorizer = None
                st.session_state.svc_model_loaded = False
    
    # Afficher l'état du chargement
    if st.session_state.get('svc_model_loaded', False):
        
        # Informations sur le modèle
        if st.session_state.svc_model and hasattr(st.session_state.svc_model, 'classes_'):
            pass
        
        if st.session_state.svc_vectorizer and hasattr(st.session_state.svc_vectorizer, 'vocabulary_'):
            pass

# Header principal
st.markdown("""
<div class="text-projection">
    <div class="front-layer">🤖 Classification de Documents Arabes</div>
    <div class="back-layer">🤖 Classification de Documents Arabes</div>
</div>
""", unsafe_allow_html=True)

# ==================== PAGES ====================
if page == "Accueil":
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📋 À propos du Système")
        st.markdown("""
        **Système de classification automatique de documents journalistiques arabes**
        
        **7 catégories de classification:**
        1. **Culture** - Arts, littérature, traditions
        2. **Finance** - Économie, marchés, affaires
        3. **Medical** - Santé, médecine, recherche
        4. **Politics** - Politique, gouvernements, relations internationales
        5. **Religion** - Croyances, pratiques religieuses
        6. **Sports** - Événements sportifs, athlètes
        7. **Tech** - Technologie, innovation, science

        """)
        
        st.markdown("""
        **💡 Comment utiliser:**
        1. Naviguez vers "Test en Temps Réel"
        2. Entrez ou téléchargez un texte arabe
        3. Cliquez sur "Lancer la Classification"
        4. Visualisez les résultats détaillés
        """)

# Page de test en temps réel
elif page == "Test en Temps Réel":
    
    # Vérifier que le système est prêt
    if not st.session_state.get('svc_model_loaded', False):
        st.error("""
        ❌ **Système non disponible.** 
        
        **Problèmes possibles:**
        1. Le fichier du modèle n'existe pas au chemin spécifié
        2. Le fichier est corrompu
        3. Les dépendances ne sont pas installées
        
        **Chemin vérifié:** `models/linear_svc.pkl`
        
        **Solution:**
        - Vérifiez que le fichier existe
        - Assurez-vous que le fichier est bien un modèle scikit-learn sauvegardé avec joblib
        """)
        st.stop()
    
    # Onglets pour différentes méthodes d'entrée
    tab1, tab2 = st.tabs(["📝 Saisie Manuelle", "📁 Téléchargement de Fichier"])
    
    text_input = ""
    
    with tab1:        
        # Options pour l'entrée
        input_option = st.radio(
            "Méthode d'entrée:",
            ["Écrire/Coller", "Utiliser un exemple"]
        )
        
        if input_option == "📝 Écrire/Coller":
            text_input = st.text_area(
                "Texte en arabe:",
                height=200,
                placeholder="أدخل النص العربي هنا...",
                help="Collez ou tapez votre texte en arabe à classifier",
                key="manual_text"
            )
        else:
            # Exemples prédéfinis pour tester différentes catégories
            example_texts = {
                "⚽ Exemple Sportif": "مباراة كرة القدم بين برشلونة وريال مدريد كانت مثيرة للغاية وانتهت بفوز برشلونة بثلاثة أهداف مقابل هدفين في دوري أبطال أوروبا",
                "💰 Exemple Financier": "ارتفع مؤشر الأسهم السعودي اليوم بنسبة 1.5% مدعوماً بصعود أسهم قطاع البنوك والصناعات الأساسية بعد إعلان النتائج المالية",
                "🏥 Exemple Médical": "اكتشف فريق من الباحثين السعوديين دواءً جديداً لعلاج مرض السكري من النوع الثاني يعتمد على تقنية النانو",
                "🏛️ Exemple Politique": "انعقد مؤتمر القمة العربية في الرياض لمناقشة القضايا السياسية والأمنية في المنطقة والعلاقات الدولية",
                "🕌 Exemple Religieux": "تتناول المحاضرة موضوع الأخلاق في الإسلام وأهمية الصدق والأمانة في المعاملات والعلاقات الاجتماعية",
                "📚 Exemple Culturel": "افتتح معرض الفنون التراثية في المتحف الوطني يعرض لوحات ومخطوطات تعود للقرن العاشر ومقتنيات أثرية نادرة",
                "💻 Exemple Technologique": "أطلقت شركة سامسونج هاتفها الذكي الجديد بشاشة قابلة للطي وتقنيات متطورة في التصوير والذكاء الاصطناعي"
            }
            
            selected_example = st.selectbox("Choisissez un exemple:", list(example_texts.keys()))
            text_input = example_texts[selected_example]
            
            # Afficher le texte choisi
            st.markdown("**Texte sélectionné:**")
            st.markdown(f'<div class="arabic-text">{text_input}</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown("""
        <div class="file-upload-box">
            <h3>📎 Télécharger un fichier texte</h3>
            <p>Formats supportés: .txt (UTF-8 encoding)</p>
            <p>Taille maximale: 10 MB</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choisir un fichier texte en arabe",
            type=['txt'],
            help="Sélectionnez un fichier texte (.txt) contenant du texte en arabe",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            try:
                # Lire le fichier texte
                text_input = uploaded_file.read().decode('utf-8')
                
                # Afficher les informations du fichier
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📄 Fichier", uploaded_file.name)
                with col2:
                    st.metric("📊 Taille", f"{uploaded_file.size / 1024:.1f} KB")
                with col3:
                    st.metric("🔤 Encodage", "UTF-8")
                
                # Afficher un aperçu
                with st.expander("👁️ Aperçu du contenu"):
                    preview = text_input[:1000] + "..." if len(text_input) > 1000 else text_input
                    st.text_area("Contenu:", preview, height=200)
                        
            except Exception as e:
                st.error(f"❌ Erreur lors de la lecture du fichier: {str(e)}")
    
    # Bouton de classification
    st.markdown("---")
    
    if st.button("Lancer la Classification avec Linear SVC", type="primary", use_container_width=True):
        if text_input and text_input.strip():
            with st.spinner("Analyse en cours avec Linear SVC..."):
                # Créer une barre de progression
                progress_bar = st.progress(0)
                
                # Étape 1: Prétraitement
                st.write("🔧 **Étape 1:** Prétraitement du texte...")
                progress_bar.progress(25)
                
                # Étape 2: Vectorisation TF-IDF
                st.write("📊 **Étape 2:** Vectorisation TF-IDF...")
                progress_bar.progress(50)
                
                # Étape 3: Classification avec Linear SVC
                st.write("🎯 **Étape 3:** Classification avec Linear SVC...")
                progress_bar.progress(75)
                
                # Classification avec le modèle
                results, predicted_category = classify_with_svc(
                    text_input, 
                    st.session_state.svc_model, 
                    st.session_state.svc_vectorizer
                )
                
                # Étape 4: Présentation des résultats
                progress_bar.progress(100)
                
                if results and predicted_category:
                    # Définir les catégories
                    if hasattr(st.session_state.svc_model, 'classes_'):
                        category_names = list(st.session_state.svc_model.classes_)
                    else:
                        category_names = ['Culture', 'Finance', 'Medical', 'Politics', 'Religion', 'Sports', 'Tech']
                    
                    # Emojis et couleurs pour les catégories
                    category_emojis = {
                        'Culture': '📚',
                        'Finance': '💰',
                        'Medical': '🏥',
                        'Politics': '🏛️',
                        'Religion': '🕌',
                        'Sports': '⚽',
                        'Tech': '💻'
                    }
                    
                    category_colors = {
                        'Culture': '#FF6B6B',
                        'Finance': '#4ECDC4',
                        'Medical': '#FFD166',
                        'Politics': '#06D6A0',
                        'Religion': '#118AB2',
                        'Sports': '#EF476F',
                        'Tech': '#7B68EE'
                    }
                    
                    # Affichage des résultats
                    st.success("✅ **Classification terminée avec succès!**")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown("#### 📊 Distribution des Probabilités")
                        
                        # Préparer les données pour le graphique
                        categories_display = [f"{category_emojis.get(cat, '📋')} {cat}" for cat in category_names]
                        probabilities = [results.get(cat, 0) for cat in category_names]
                        
                        # Créer un DataFrame pour le graphique
                        df_results = pd.DataFrame({
                            'Catégorie': categories_display,
                            'Probabilité': probabilities
                        })
                        
                        # Trier par probabilité
                        df_results = df_results.sort_values('Probabilité', ascending=False)
                        
                        # Graphique à barres
                        fig = px.bar(
                            df_results,
                            x='Catégorie',
                            y='Probabilité',
                            color='Probabilité',
                            color_continuous_scale='oranges',
                            title='Distribution des Scores par Catégorie'
                        )
                        fig.update_layout(yaxis_range=[0, 1], showlegend=False)
                        fig.update_yaxes(tickformat=".0%", title="Probabilité")
                        fig.update_xaxes(title="Catégorie")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Tableau détaillé des scores
                        st.markdown("#### 📋 Scores Détailés")
                        
                        for cat in category_names:
                            score = results.get(cat, 0)
                            emoji = category_emojis.get(cat, '📋')
                            
                            col_a, col_b, col_c = st.columns([1, 6, 2])
                            with col_a:
                                st.write(f"**{emoji}**")
                            with col_b:
                                st.progress(float(score))
                            with col_c:
                                st.write(f"**{score*100:.1f}%**")
                    
                    with col2:
                        st.markdown("#### 🏆 Résultat de Classification")
                        
                        # Récupérer l'emoji et la couleur pour la catégorie prédite
                        pred_emoji = category_emojis.get(predicted_category, '🎯')
                        pred_color = category_colors.get(predicted_category, '#4A90E2')
                        
                        # Afficher la carte de résultat
                        st.markdown(f"""
                        <div style="background: {pred_color}; padding: 2rem; border-radius: 15px; color: white; text-align: center;">
                            <h2>{pred_emoji} {predicted_category}</h2>
                            <h1 style="font-size: 3rem; margin: 1rem 0;">{results[predicted_category]*100:.1f}%</h1>
                            <p>Confiance de prédiction</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Métriques clés
                        st.metric("🎯 Catégorie", f"{pred_emoji} {predicted_category}")
                        st.metric("📊 Confiance", f"{results[predicted_category]*100:.1f}%")
                        
                        # Calculer la marge avec la deuxième catégorie
                        sorted_scores = sorted(results.items(), key=lambda x: x[1], reverse=True)
                        if len(sorted_scores) > 1:
                            margin = sorted_scores[0][1] - sorted_scores[1][1]
                            st.metric("📈 Marge", f"{margin*100:.1f}%")
                        
                        # Information technique
                        st.markdown('<div class="svc-highlight">', unsafe_allow_html=True)
                        st.write("**⚙️ Modèle utilisé:** Linear Support Vector Classifier (SVC)")
                        if hasattr(st.session_state.svc_model, 'coef_'):
                            st.write(f"**🔢 Features:** {st.session_state.svc_model.coef_.shape[1]}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Section de détails techniques
                    with st.expander("🔍 Détails Techniques et Analyse"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            for stat, value in stats.items():
                                st.write(f"• **{stat}:** {value}")
                            
                            st.markdown("**🔧 Étapes de Prétraitement:**")
                            st.write("1. Normalisation des caractères arabes")
                            st.write("2. Suppression des diacritiques (tashkeel)")
                            st.write("3. Élimination de la ponctuation")
                            st.write("4. Suppression des chiffres")
                            st.write("5. Filtrage des stopwords arabes")
                            st.write("6. Normalisation des espaces")
                        
                        with col2:
                            st.markdown("**🎯 Analyse des Scores:**")
                            
                            # Top 3 catégories
                            top_3 = sorted(results.items(), key=lambda x: x[1], reverse=True)[:3]
                            for i, (cat, score) in enumerate(top_3):
                                emoji = category_emojis.get(cat, '📋')
                                st.write(f"{i+1}. {emoji} **{cat}:** {score:.1%}")
                            
                            # Niveau de confiance
                            confidence = results[predicted_category]
                            if confidence > 0.7:
                                st.write("• 🟢 **Confiance élevée** (supérieure à 70%)")
                            elif confidence > 0.5:
                                st.write("• 🟡 **Confiance moyenne** (entre 50% et 70%)")
                            else:
                                st.write("• 🔴 **Confiance faible** (inférieure à 50%)")
                            
                            # Informations sur le modèle entraîné
                            if hasattr(st.session_state.svc_model, 'n_iter_'):
                                st.write(f"• **Itérations:** {st.session_state.svc_model.n_iter_}")
            
                else:
                    st.error("❌ **Échec de la classification**")
                    st.info("""
                    **Solutions possibles:**
                    1. Vérifiez que le texte contient suffisamment de mots (au moins 5-10 mots)
                    2. Assurez-vous que le texte est en arabe
                    3. Vérifiez l'encodage du texte (UTF-8 recommandé)
                    4. Essayez avec un exemple prédéfini pour tester le système
                    """)
        else:
            st.warning("⚠️ **Veuillez entrer ou télécharger un texte à classifier**")

# Footer
st.markdown("""
<div class="footer">
    <div class="footer-brand">SNI TASNEEF</div>
    <p><strong>Système de Classification Intelligente de Documents Arabes</strong></p>
    <p>© 2025</p>
</div>
""", unsafe_allow_html=True)