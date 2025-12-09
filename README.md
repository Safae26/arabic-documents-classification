# 🤖 Classification de Documents Arabes

Application web de classification automatique de documents journalistiques arabes utilisant Linear Support Vector Classifier.

## 🚀 Fonctionnalités

- **Classification en 7 catégories** : Culture, Finance, Medical, Politics, Religion, Sports, Tech
- **Interface Streamlit** intuitive et responsive
- **Prétraitement avancé** du texte arabe
- **Visualisations interactives** avec Plotly
- **Support pour fichiers texte** et saisie manuelle

## 📦 Installation

### Prérequis
- Python 3.8+
- pip ou conda

### Installation des dépendances

```bash
# Cloner le projet
git clone https://github.com/votre-username/mon-projet-classification.git
cd mon-projet-classification

# Créer un environnement virtuel (optionnel)
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt

# Télécharger les ressources NLTK
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### Référence : https://www.researchgate.net/publication/359704038_An_Effective_Approach_for_Arabic_Document_Classification_Using_Machine_Learning