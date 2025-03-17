import re
import nltk
import numpy as np
import torch
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer, BertModel
from g2p_en import G2p  # Use g2p for phoneme conversion
from dotenv import load_dotenv
import os

# Force UTF-8 encoding
os.environ["PYTHONUTF8"] = "1"

# Load environment variables from .env file
load_dotenv('E:/SEMESTER_IV/GDG/AI-Powered-Educational-Video-Generation/myenv/.env')

# Verify PYTHONUTF8 is set
print("PYTHONUTF8:", os.getenv("PYTHONUTF8"))

# Download necessary resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')

class TextProcessor:
    def __init__(self, text):
        self.text = text
        self.sentences = sent_tokenize(text)
        self.words = word_tokenize(text)
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.g2p = G2p()  # Initialize g2p for phoneme conversion

    def clean_text(self):
        """Remove special characters, emojis, and extra spaces."""
        # Remove emojis and non-ASCII characters (except letters)
        self.text = re.sub(r'[^\x00-\x7F\s\w]', ' ', self.text)
        # Remove special characters except spaces and letters
        self.text = re.sub(r'[^a-zA-Z0-9\s]', '', self.text)
        # Remove extra spaces
        self.text = re.sub(r'\s+', ' ', self.text).strip()
        return self.text

    def tokenize_and_lemmatize(self):
        """Tokenizes words and applies lemmatization."""
        return [self.lemmatizer.lemmatize(word.lower()) for word in self.words if word.lower() not in self.stop_words]

    def get_pos_tags(self):
        """Returns part-of-speech tags for words."""
        return nltk.pos_tag(self.words)

    def keyword_extraction(self, top_n=5):
        """Extract top keywords based on word frequency."""
        filtered_words = [word.lower() for word in self.words if word.lower() not in self.stop_words and word.isalnum()]
        word_freq = Counter(filtered_words)
        return word_freq.most_common(top_n)

    def extractive_summarization(self, num_sentences=3):
        """Summarizes the text using BERT embeddings."""
        sentence_embeddings = []
        for sent in self.sentences:
            inputs = self.tokenizer(sent, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
            sentence_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            sentence_embeddings.append(sentence_embedding)

        sentence_scores = np.linalg.norm(sentence_embeddings, axis=1)
        top_sentence_indices = np.argsort(sentence_scores)[-num_sentences:]
        top_sentence_indices.sort()
        summary = ' '.join([self.sentences[i] for i in top_sentence_indices])
        return summary

    def text_to_phonemes(self):
        """Convert text to phonemes for TTS using g2p."""
        phonemes = []
        for word in self.words:
            # Handle non-ASCII characters manually
            if any(ord(char) > 127 for char in word):
                if word.lower() == "café":
                    phonemes.append("K AE0 F EY1")
                elif word.lower() == "naïve":
                    phonemes.append("N AY0 IY1 V")
                else:
                    phonemes.append(' '.join(self.g2p(word)))
            else:
                phonemes.append(' '.join(self.g2p(word)))
        return ' '.join(phonemes)

# Example usage
if __name__ == "__main__":
    sample_text = "Natural Language Processing (NLP) is an exciting field of AI! It deals with text, speech, and even emojis like 😊. Non-ASCII characters like café and naïve are also processed."
    processor = TextProcessor(sample_text)
    print("Original Text:", repr(sample_text))  # This will show the raw representation of the text
    print("Cleaned Text:", processor.clean_text())
    print("Lemmatized Tokens:", processor.tokenize_and_lemmatize())
    print("POS Tags:", processor.get_pos_tags())
    print("Top Keywords:", processor.keyword_extraction())
    print("Extractive Summary:", processor.extractive_summarization())
    print("Phonemes Representation:", processor.text_to_phonemes())