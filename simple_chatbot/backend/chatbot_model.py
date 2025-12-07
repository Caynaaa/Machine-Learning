"""
chatbot_model.py
This file just contains a placeholder for the chatbot model loading function.
like a (load model, preprocessing, prediction)
"""

import torch
import torch.nn as nn
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import random
import os


try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
        nltk.data.find('corpora/wordnet')
except LookupError:
        print("Downloading NLTK data...")
        nltk.download('punkt')
        nltk.download('punkt_tab')   
        nltk.download('wordnet')

# Model Architectur
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Model Loader
class ChatBotModel:
    def __init__(self):
        self.model_path = None
        self.model = None
        self.all_words = None
        self.tags = None
        self.intents = None
        self.lemmatizer = WordNetLemmatizer()
        self.is_loaded = False
        
        # get model path
        if self.model_path is None:
            current_dir = os.path.dirname(__file__) # get name folder chatbot_mdoel.py
            self.model_path = os.path.join(current_dir, "simple_chatbot_model.pth")
        else:
            self.model_path = self.model_path
    
    # load based model and all weights
    def load(self):
        print(f"Loading Chatbot Model from {self.model_path}...")
        
        try:
            # load model data
            model_data = torch.load(self.model_path, map_location=torch.device('cpu'))
            
            # re-create model
            self.model = SimpleNN(
                model_data["input_size"],
                model_data["hidden_size"],
                model_data["output_size"]
            )
            
            # laod weights
            self.model.load_state_dict(model_data["model_state"])
            # make sure the model in eval mode
            self.model.eval()

            # Store vocalbulary and intents
            self.all_words = model_data["all_words"]
            self.tags = model_data["tags"]
            self.intents = model_data["intents"]
            self.is_loaded = True
            
            print("Model berhasil diload!")
            print(f"    - Intents: {len(self.tags)}")
            print(f"    - Vocalbulary size: {len(self.all_words)} kata")
        
        except FileNotFoundError:
            raise Exception(f"Model '{self.model_path}' not found.")
        except Exception as e:
            raise Exception(f"Failed to laod model: {str(e)}")
    
    # Prediction input user
    def predict(self, sentence):
        if not self.is_loaded:
            raise Exception("Model not loaded. Call load() first")
        
        # preprocess input
        words = nltk.word_tokenize(sentence)
        words = [self.lemmatizer.lemmatize(w.lower()) for w in words]
        
        # bag of words
        bag = [1 if word in words else 0 for word in self.all_words]
        bag = np.array(bag, dtype=np.float32).reshape(1, -1)
        
        # Predict
        with torch.no_grad():
            input_tensor = torch.FloatTensor(bag)
            output = self.model(input_tensor)
            probabilitys = torch.softmax(output, dim=1)
            prob, predicted = torch.max(probabilitys, 1)
            
        intent = self.tags[predicted.item()]
        confidence = prob.item()
        
        return intent, confidence
    
    # Check confidence level
    def get_confidence_level(self, confidence):
        if confidence >= 0.8:
            return 'high'
        elif confidence >= 0.6:
            return 'medium'
        elif confidence >= 0.4:
            return 'low'
        
    # Get response with confidence check
    def get_response_with_warning(self, sentence):
        intent, confidence = self.predict(sentence)
        confidence_level = self.get_confidence_level(confidence)
        
        # Get based response
        base_response = self.get_response(intent)
        
        # generate warning based on confidence
        warning = "" # -> default with no warning
        if confidence_level == 'low':
            warning = f"Saya tidak yakin dengan jawaban ini. Silahkan coba guankan kata lain."
        elif confidence_level == 'medium':
            warning = f"Saya ragu dengan jawaban ini."
        
        return {
            'intent': intent,
            'confidence': confidence,
            'confidence_level': confidence_level,
            'response': base_response,
            'warning': warning,
            'has_warning': bool(warning)
        }
    
    # Respon based on intent
    def get_response(self, intent_tag):
        for intent in self.intents:
            if intent['tag'] == intent_tag:
                return random.choice(intent['response'])
        return "Maaf, saya tidak mengerti."


# global instance can be acces in all file app.py
# so that model only load once at the start of app
_chatbot_instance = None
def get_chatbot():
    global _chatbot_instance
    if _chatbot_instance is None:
        _chatbot_instance = ChatBotModel()
    return _chatbot_instance