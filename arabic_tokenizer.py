from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

class ArabicTokenizer:
    
    def __init__(self):
        self.stop_words = set(stopwords.words('arabic')) 
    
    def tokenize(self, text, remove_stopwords=True):
        # Tokenise le texte en mots
        sentences = sent_tokenize(text)

        all_tokens = []
        for sentence in sentences:
            words = word_tokenize(sentence)
            
            if remove_stopwords:
                words = [word for word in words if word not in self.stop_words]

            words = [str(word) for word in words]
            all_tokens.extend(words)
        
        return all_tokens