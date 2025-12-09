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
    /* ===== VARIABLES DE COULEUR - THÈME FUTURISTE BLEU CYBER ===== */
    :root {
        --primary: #0EA5E9;
        --primary-dark: #0369A1;
        --primary-light: #38BDF8;
        --primary-50: #F0F9FF;
        --primary-100: #E0F2FE;
        --primary-200: #BAE6FD;
        --primary-300: #7DD3FC;
        --primary-400: #38BDF8;
        --primary-500: #0EA5E9;
        --primary-600: #0284C7;
        --primary-700: #0369A1;
        --primary-800: #075985;
        --primary-900: #0C4A6E;
        
        --cyber-blue: #00D4FF;
        --cyber-purple: #8B5CF6;
        --cyber-pink: #EC4899;
        --neon-blue: #00F7FF;
        
        --dark-bg: #0F172A;
        --darker-bg: #020617;
        --card-bg: rgba(30, 41, 59, 0.7);
        --card-bg-light: rgba(51, 65, 85, 0.8);
        --card-border: rgba(71, 85, 105, 0.3);
        --card-border-light: rgba(100, 116, 139, 0.5);
        
        --text-primary: #FFFFFF;
        --text-secondary: #E2E8F0;
        --text-muted: #94A3B8;
        --text-glow: rgba(255, 255, 255, 0.9);
        
        --gradient-blue: linear-gradient(135deg, var(--primary-500) 0%, var(--cyber-blue) 100%);
        --gradient-cyber: linear-gradient(135deg, var(--cyber-blue) 0%, var(--cyber-purple) 50%, var(--cyber-pink) 100%);
        --gradient-dark: linear-gradient(135deg, var(--darker-bg) 0%, var(--dark-bg) 100%);
        --gradient-glass: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
        
        --shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        --shadow-lg: 0 20px 40px rgba(0, 0, 0, 0.4);
        --shadow-xl: 0 30px 60px rgba(0, 0, 0, 0.5);
        --shadow-neon: 0 0 20px rgba(14, 165, 233, 0.3);
        
        --border-radius: 16px;
        --border-radius-lg: 24px;
        --border-radius-sm: 12px;
    }
    
    /* ===== ANIMATIONS ===== */
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    @keyframes glow {
        0%, 100% { box-shadow: var(--shadow-neon); }
        50% { box-shadow: 0 0 30px rgba(14, 165, 233, 0.5); }
    }
    
    @keyframes shimmer {
        0% { background-position: -200% center; }
        100% { background-position: 200% center; }
    }
    
    @keyframes slideIn {
        from { transform: translateY(20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    /* ===== FOND GÉNÉRAL AVEC EFFET DE PARTICULES ===== */
    .stApp {
        background: var(--darker-bg);
        background-image: 
            radial-gradient(circle at 10% 20%, rgba(14, 165, 233, 0.1) 0%, transparent 20%),
            radial-gradient(circle at 90% 80%, rgba(139, 92, 246, 0.1) 0%, transparent 20%),
            radial-gradient(circle at 50% 50%, rgba(0, 212, 255, 0.05) 0%, transparent 30%);
        color: var(--text-primary);
        animation: slideIn 0.6s ease-out;
    }
    
    /* ===== TOUS LES TEXTES ===== */
    * {
        color: var(--text-primary) !important;
        font-smooth: always;
        -webkit-font-smoothing: antialiased;
    }
    
    /* ===== EN-TÊTE PRINCIPAL CYBER ===== */
    .main-header {
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, var(--text-primary) 0%, var(--primary-300) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        padding: 3rem;
        margin-bottom: 3rem;
        background: rgba(30, 41, 59, 0.3);
        backdrop-filter: blur(10px);
        border: 1px solid var(--card-border);
        border-radius: var(--border-radius-lg);
        position: relative;
        overflow: hidden;
        animation: float 6s ease-in-out infinite;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: var(--gradient-cyber);
        z-index: -1;
        border-radius: var(--border-radius-lg);
        opacity: 0.3;
    }
    
    .main-header::after {
        content: '🤖 SNI TASNEEF';
        font-size: 1.2rem;
        color: var(--cyber-blue);
        display: block;
        margin-top: 0.8rem;
        font-weight: 600;
        letter-spacing: 4px;
        text-transform: uppercase;
        background: linear-gradient(90deg, var(--cyber-blue), var(--cyber-purple));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* ===== SOUS-TITRES HOLOGRAFIQUES ===== */
    .sub-header {
        font-size: 2.2rem;
        background: linear-gradient(90deg, var(--primary-400), var(--cyber-blue));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        border-bottom: 2px solid;
        border-image: linear-gradient(90deg, var(--primary-500), transparent) 1;
        padding-bottom: 1.2rem;
        margin-top: 3rem;
        margin-bottom: 2rem;
        font-weight: 800;
        position: relative;
        padding-left: 0;
        text-shadow: 0 2px 10px rgba(14, 165, 233, 0.3);
    }
    
    .sub-header::after {
        content: '';
        position: absolute;
        bottom: -2px;
        left: 0;
        width: 100px;
        height: 4px;
        background: var(--gradient-blue);
        border-radius: 2px;
    }
    
    /* ===== CARTES MÉTRIQUES GLASSMORPHISM ===== */
    .metric-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: var(--border-radius);
        border: 1px solid var(--card-border);
        text-align: center;
        box-shadow: var(--shadow);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        min-height: 150px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        position: relative;
        overflow: hidden;
        animation: glow 3s ease-in-out infinite;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transition: 0.5s;
    }
    
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: var(--shadow-xl), 0 0 40px rgba(14, 165, 233, 0.4);
        border-color: var(--primary-500);
        background: rgba(30, 41, 59, 0.6);
    }
    
    .metric-card:hover::before {
        left: 100%;
    }
    
    .metric-card .metric-value {
        font-size: 3rem;
        font-weight: 900;
        background: var(--gradient-blue);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0;
        position: relative;
        z-index: 1;
    }
    
    .metric-card .metric-label {
        font-size: 0.9rem;
        color: var(--text-secondary) !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 600;
        opacity: 0.9;
    }
    
    /* ===== CARTE DE RÉSULTAT CYBER ===== */
    .result-card {
        background: rgba(14, 165, 233, 0.1);
        backdrop-filter: blur(15px);
        padding: 3rem;
        border-radius: var(--border-radius-lg);
        border: 2px solid rgba(14, 165, 233, 0.3);
        text-align: center;
        box-shadow: var(--shadow-lg), inset 0 0 50px rgba(14, 165, 233, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .result-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: var(--gradient-cyber);
        opacity: 0.1;
        z-index: -1;
    }
    
    .result-card::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: conic-gradient(
            transparent,
            var(--cyber-blue),
            transparent 30%
        );
        animation: rotate 4s linear infinite;
    }
    
    /* ===== ZONE DE TÉLÉCHARGEMENT CYBER ===== */
    .file-upload-box {
        border: 2px dashed;
        border-image: linear-gradient(135deg, var(--primary-500), var(--cyber-blue)) 1;
        border-radius: var(--border-radius);
        padding: 4rem 2rem;
        text-align: center;
        background: rgba(14, 165, 233, 0.05);
        backdrop-filter: blur(5px);
        margin: 2.5rem 0;
        transition: all 0.3s ease;
    }
    
    .file-upload-box:hover {
        background: rgba(14, 165, 233, 0.1);
        box-shadow: var(--shadow-lg), 0 0 30px rgba(14, 165, 233, 0.2);
        transform: scale(1.01);
    }
    
    .file-upload-box h3 {
        background: var(--gradient-blue);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.8rem;
        margin-bottom: 1rem;
    }
    
    /* ===== TEXTE ARABE FUTURISTE ===== */
    .arabic-text {
        direction: rtl;
        text-align: right;
        font-size: 1.5em;
        line-height: 2;
        padding: 2.5rem;
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(10px);
        border-radius: var(--border-radius);
        border-right: 4px solid var(--primary-500);
        font-family: 'Segoe UI', system-ui, sans-serif;
        box-shadow: var(--shadow);
        margin: 2rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .arabic-text::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: var(--gradient-blue);
    }
    
    /* ===== SIDEBAR CYBER ===== */
    [data-testid="stSidebar"] {
        background: rgba(2, 6, 23, 0.8) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(14, 165, 233, 0.2);
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background: rgba(14, 165, 233, 0.1);
        padding: 2.5rem 1.5rem;
        border-bottom: 1px solid rgba(14, 165, 233, 0.2);
    }
    
    .sidebar-title {
        font-size: 2rem;
        font-weight: 900;
        text-align: center;
        position: relative;
        padding-bottom: 1.5rem;
        background: var(--gradient-cyber);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2.5rem;
    }
    
    .sidebar-title::after {
        content: '';
        display: block;
        width: 80px;
        height: 4px;
        background: var(--gradient-cyber);
        margin: 15px auto;
        border-radius: 2px;
        animation: glow 2s ease-in-out infinite;
    }
    
    /* ===== BOUTONS CYBER ===== */
    .stButton > button {
        background: var(--gradient-blue);
        color: white !important;
        border: none;
        padding: 1.2rem 3rem;
        border-radius: 12px;
        font-weight: 700;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: var(--shadow);
        font-size: 1.1rem;
        position: relative;
        overflow: hidden;
        letter-spacing: 0.5px;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: 0.5s;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: var(--shadow-lg), 0 0 30px rgba(14, 165, 233, 0.4);
        background: var(--gradient-blue);
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button[kind="primary"] {
        background: var(--gradient-cyber);
        border: 2px solid rgba(255, 255, 255, 0.2);
        font-weight: 800;
        animation: glow 2s ease-in-out infinite;
    }
    
    /* ===== ONGLETS CYBER ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(10px);
        padding: 0.8rem;
        border-radius: var(--border-radius-sm);
        border: 1px solid var(--card-border);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(14, 165, 233, 0.1);
        border-radius: 10px;
        padding: 1.2rem 2.5rem;
        border: 1px solid var(--card-border);
        font-weight: 600;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(14, 165, 233, 0.2);
        border-color: var(--primary-500);
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: var(--gradient-blue);
        border-color: var(--primary-500);
        box-shadow: var(--shadow), 0 0 20px rgba(14, 165, 233, 0.3);
        transform: scale(1.05);
    }
    
    /* ===== INPUTS CYBER ===== */
    .stTextArea textarea, 
    .stTextInput input,
    .stSelectbox div[data-baseweb="select"] {
        background: rgba(30, 41, 59, 0.6) !important;
        backdrop-filter: blur(5px);
        border: 2px solid rgba(71, 85, 105, 0.5) !important;
        border-radius: 12px !important;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    
    .stTextArea textarea:focus,
    .stTextInput input:focus {
        border-color: var(--primary-500) !important;
        box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.2) !important;
        background: rgba(30, 41, 59, 0.8) !important;
    }
    
    /* ===== PROGRESS BAR CYBER ===== */
    .stProgress > div > div > div {
        background: var(--gradient-cyber);
        border-radius: 10px;
    }
    
    /* ===== EXPANDER CYBER ===== */
    .streamlit-expanderHeader {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(10px);
        border: 2px solid var(--card-border);
        border-radius: 12px;
        font-weight: 700;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(14, 165, 233, 0.2);
        border-color: var(--primary-500);
    }
    
    .streamlit-expanderContent {
        background: rgba(2, 6, 23, 0.8);
        border: 2px solid var(--card-border);
        border-radius: 0 0 12px 12px;
        margin-top: -2px;
        backdrop-filter: blur(10px);
    }
    
    /* ===== ALERTES CYBER ===== */
    .stAlert {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border: 2px solid;
        border-radius: var(--border-radius);
    }
    
    .stAlert.success {
        border-color: var(--primary-500);
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.1), rgba(30, 41, 59, 0.7));
    }
    
    .stAlert.error {
        border-color: #EF4444;
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(30, 41, 59, 0.7));
    }
    
    .stAlert.warning {
        border-color: #F59E0B;
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(30, 41, 59, 0.7));
    }
    
    .stAlert.info {
        border-color: var(--primary-400);
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.1), rgba(30, 41, 59, 0.7));
    }
    
    /* ===== FOOTER CYBER ===== */
    .footer {
        text-align: center;
        padding: 3rem;
        font-size: 0.95rem;
        border-top: 1px solid rgba(14, 165, 233, 0.2);
        margin-top: 4rem;
        background: rgba(14, 165, 233, 0.05);
        backdrop-filter: blur(10px);
        border-radius: var(--border-radius-lg);
        position: relative;
        overflow: hidden;
    }
    
    .footer::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: var(--gradient-cyber);
    }
    
    .footer-brand {
        font-size: 2rem;
        font-weight: 900;
        margin-bottom: 1rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        background: var(--gradient-cyber);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .footer-subtitle {
        color: var(--text-secondary) !important;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
        opacity: 0.9;
    }
    
    /* ===== SCROLLBAR CYBER ===== */
    ::-webkit-scrollbar {
        width: 12px;
        height: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(14, 165, 233, 0.1);
        border-radius: 6px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--gradient-blue);
        border-radius: 6px;
        border: 2px solid var(--darker-bg);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--gradient-cyber);
    }
    
    /* ===== EFFETS DE SURBRILANCE ===== */
    .highlight-text {
        background: linear-gradient(90deg, var(--cyber-blue), var(--cyber-purple));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    .glow-effect {
        filter: drop-shadow(0 0 10px rgba(14, 165, 233, 0.5));
    }
    
    /* ===== GRADIENTS ANIMÉS ===== */
    .animated-gradient {
        background: linear-gradient(-45deg, var(--primary-500), var(--cyber-blue), var(--cyber-purple), var(--cyber-pink));
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* ===== RESPONSIVE ===== */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
            padding: 2rem;
        }
        
        .sub-header {
            font-size: 1.8rem;
        }
        
        .metric-card .metric-value {
            font-size: 2.5rem;
        }
        
        .file-upload-box {
            padding: 3rem 1.5rem;
        }
        
        .stButton > button {
            padding: 1rem 2rem;
            font-size: 1rem;
        }
    }
    
    /* ===== EFFETS SPÉCIAUX POUR GRAPHIQUES ===== */
    .js-plotly-plot {
        background: rgba(30, 41, 59, 0.5) !important;
        backdrop-filter: blur(10px);
        border-radius: var(--border-radius);
        padding: 1.5rem;
        border: 1px solid var(--card-border);
    }
    
    /* ===== SPINNER CYBER ===== */
    .stSpinner > div {
        border-color: var(--cyber-blue) transparent transparent transparent;
    }
    
    /* ===== BADGES DE CATÉGORIES ===== */
    .category-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        background: rgba(14, 165, 233, 0.1);
        border: 1px solid rgba(14, 165, 233, 0.3);
        border-radius: 20px;
        font-weight: 600;
        margin: 0.3rem;
        transition: all 0.3s ease;
    }
    
    .category-badge:hover {
        background: rgba(14, 165, 233, 0.2);
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(14, 165, 233, 0.2);
    }
    
    /* ===== EFFETS DE LUMIÈRE ===== */
    .light-effect {
        position: fixed;
        pointer-events: none;
        z-index: -1;
    }
    
    .light-effect:nth-child(1) {
        top: 20%;
        left: 10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(14, 165, 233, 0.1) 0%, transparent 70%);
    }
    
    .light-effect:nth-child(2) {
        bottom: 20%;
        right: 10%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(139, 92, 246, 0.1) 0%, transparent 70%);
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
                st.success("✅ Modèle chargé")
            else:
                st.error("❌ Échec du chargement")
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
st.markdown('<h1 class="main-header">🤖 Classification de Documents Arabes</h1>', unsafe_allow_html=True)

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
            ["📝 Écrire/Coller", "🔍 Utiliser un exemple"]
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
                            st.markdown("**📊 Statistiques du Texte:**")
                            
                            # Calculer les statistiques
                            original_words = text_input.split()
                            cleaned_text = arabic_preprocessing(text_input)
                            cleaned_words = cleaned_text.split()
                            
                            stats = {
                                "Mots originaux": len(original_words),
                                "Caractères originaux": len(text_input),
                                "Mots après prétraitement": len(cleaned_words),
                                "Mots uniques": len(set(cleaned_words)),
                                "Taux de réduction": f"{(len(original_words) - len(cleaned_words))/max(len(original_words), 1)*100:.1f}%"
                            }
                            
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
                            
                            st.markdown("**⚙️ Configuration Linear SVC:**")
                            st.write("• **Algorithme:** Support Vector Machine linéaire")
                            st.write("• **Kernel:** Linéaire (pas de transformation kernel)")
                            st.write("• **Perte:** Hinge loss")
                            st.write("• **Régularisation:** L2")
                            
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