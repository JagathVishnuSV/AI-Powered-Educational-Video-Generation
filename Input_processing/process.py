import re
import nltk
import numpy as np
import torch
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer, BertModel
import epitran


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
        self.epi = epitran.Epitran('eng-Latn')
    
    def clean_text(self):
        """Remove special characters and extra spaces."""
        self.text = re.sub(r'[^\x00-\x7F]+', ' ', self.text) 
        self.text = re.sub(r'[^a-zA-Z0-9\s]', '', self.text)
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
        """Convert text to phonemes for TTS"""
        return ' '.join([self.epi.transliterate(word) for word in self.words])

# Example usage
if __name__ == "__main__":
    sample_text = "Natural Language Processing is an exciting field of AI that deals with text and speech processing. It enables machines to understand human language. NLP has various applications, including chatbots, voice assistants, and automated translations. The advancements in deep learning have improved NLP significantly. Companies use NLP for customer support and analytics. AI-driven text generation is becoming popular."
    processor = TextProcessor(sample_text)
    print("Original Text:", repr(sample_text))  # This will show the raw representation of the text
    print("Cleaned Text:", processor.clean_text())
    print("Lemmatized Tokens:", processor.tokenize_and_lemmatize())
    print("POS Tags:", processor.get_pos_tags())
    print("Top Keywords:", processor.keyword_extraction())
    print("Extractive Summary:", processor.extractive_summarization())
    print("Phonemes Reperesentation:", processor.text_to_phonemes())

   
