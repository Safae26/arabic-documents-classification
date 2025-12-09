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
    page_title="Classification de Documents Arabes - Linear SVC",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #4A90E2;
        border-bottom: 2px solid #4A90E2;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .file-upload-box {
        border: 2px dashed #4A90E2;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9fa;
        margin: 1rem 0;
    }
    .result-card {
        background: linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
    }
    .arabic-text {
        direction: rtl;
        text-align: right;
        font-size: 1.2em;
        line-height: 1.8;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 10px;
        border-right: 4px solid #4A90E2;
    }
    .svc-highlight {
        background: linear-gradient(135deg, #FF8C00 0%, #FFA500 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
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
        
        st.success(f"✅ Modèle LinearSVC chargé avec succès")
        
        # Afficher les informations du modèle
        if hasattr(model, 'classes_'):
            st.info(f"📊 Catégories: {len(model.classes_)}")
        
        if hasattr(model, 'coef_'):
            st.info(f"🔢 Nombre de features: {model.coef_.shape[1]}")
        
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
    st.markdown("## 🤖 CLASSIFICATEUR LINEAR SVC")
    st.markdown("---")
    
    # Navigation
    page = st.radio(
        "Navigation:",
        ["Accueil", "Test en Temps Réel"]
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
        st.success("✅ Système prêt")
        
        # Informations sur le modèle
        if st.session_state.svc_model and hasattr(st.session_state.svc_model, 'classes_'):
            st.info(f"🎯 {len(st.session_state.svc_model.classes_)} catégories")
        
        if st.session_state.svc_vectorizer and hasattr(st.session_state.svc_vectorizer, 'vocabulary_'):
            st.info(f"📚 {len(st.session_state.svc_vectorizer.vocabulary_)} mots")

# Header principal
st.markdown('<h1 class="main-header">🤖 Classification de Documents Arabes - Linear SVC</h1>', unsafe_allow_html=True)

# ==================== PAGES ====================
if page == "Accueil":
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📋 À propos du Système")
        st.markdown("""
        **Système de classification automatique de documents journalistiques arabes**
        
        **🎯 Modèle utilisé: Linear Support Vector Classifier (SVC)**
        
        **Caractéristiques techniques:**
        - ⚡ **Algorithme:** Linear SVC (Support Vector Machine linéaire)
        - 🔧 **Prétraitement:** Normalisation arabe avancée
        - 📊 **Vectorisation:** TF-IDF (Term Frequency-Inverse Document Frequency)
        - 🎯 **Précision:** Modèle optimisé pour la classification multiclasse
        
        **7 catégories de classification:**
        1. **Culture** - Arts, littérature, traditions
        2. **Finance** - Économie, marchés, affaires
        3. **Medical** - Santé, médecine, recherche
        4. **Politics** - Politique, gouvernements, relations internationales
        5. **Religion** - Croyances, pratiques religieuses
        6. **Sports** - Événements sportifs, athlètes
        7. **Tech** - Technologie, innovation, science
        
        **Statut du système:**
        """)
        
        # Afficher l'état du système
        if st.session_state.get('svc_model_loaded', False):
            st.success("✅ **Système opérationnel** - Prêt à classifier")
        else:
            st.error("❌ **Système non disponible**")
        
        st.markdown("""
        **💡 Comment utiliser:**
        1. Naviguez vers "Test en Temps Réel"
        2. Entrez ou téléchargez un texte arabe
        3. Cliquez sur "Lancer la Classification"
        4. Visualisez les résultats détaillés
        """)
    
    with col2:
        st.markdown("#### 📊 Informations Techniques")
        
        if st.session_state.get('svc_model_loaded', False):
            info_cards = []
            
            # Informations sur le modèle
            if hasattr(st.session_state.svc_model, 'classes_'):
                info_cards.append(("🎯", "Catégories", str(len(st.session_state.svc_model.classes_))))
            
            if hasattr(st.session_state.svc_vectorizer, 'vocabulary_'):
                info_cards.append(("📚", "Mots uniques", str(len(st.session_state.svc_vectorizer.vocabulary_))))
            
            if hasattr(st.session_state.svc_model, 'coef_'):
                info_cards.append(("🔢", "Features", str(st.session_state.svc_model.coef_.shape[1])))
            
            info_cards.append(("⚡", "Algorithme", "Linear SVC"))
            info_cards.append(("🔤", "Langue", "Arabe"))
            info_cards.append(("🔄", "Prétraitement", "Complet"))
            
            for emoji, label, value in info_cards:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                            padding: 1rem; border-radius: 10px; margin: 0.5rem 0; text-align: center;">
                    <div style="font-size: 2rem;">{emoji}</div>
                    <div><strong>{label}</strong></div>
                    <div style="font-size: 1.5rem; font-weight: bold;">{value}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("ℹ️ Modèle non chargé. Veuillez vérifier le chemin du fichier.")

# Page de test en temps réel
elif page == "Test en Temps Réel":
    st.markdown('<h2 class="sub-header">🔍 Test en Temps Réel - Linear SVC</h2>', unsafe_allow_html=True)
    
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
    
    st.success("✅ **Système Linear SVC prêt** - Entrez un texte arabe pour classification")
    
    # Onglets pour différentes méthodes d'entrée
    tab1, tab2 = st.tabs(["📝 Saisie Manuelle", "📁 Téléchargement de Fichier"])
    
    text_input = ""
    
    with tab1:        
        # Options pour l'entrée
        input_option = st.radio(
            "Méthode d'entrée:",
            ["📝 Écrire/Coller", "🎯 Utiliser un exemple"]
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
    
    if st.button("🚀 Lancer la Classification avec Linear SVC", type="primary", use_container_width=True):
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
                            title='Distribution des Scores par Catégorie - Linear SVC'
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
---
<div style="text-align: center; color: #666; font-size: 0.9em;">
<p>📚 **Système de Classification de Documents Arabes** - Linear Support Vector Classifier (SVC)</p>
<p>Développé pour la classification automatique de documents journalistiques arabes</p>
<p>© 2024 - Tous droits réservés</p>
</div>
""", unsafe_allow_html=True)