"""
Campus ChatBot with ML
"""


# import library
import torch
import torch.nn as nn
import numpy as np
import json
import nltk
from nltk.stem import WordNetLemmatizer
import random
import sys


# Initiliaze
lemmatizer = WordNetLemmatizer()

# --- Load Model ---
def load_model(model_path="./simple_chatbot_model.pth"):
    print("Load campus cahtbot...")
    
    # load model data
    model_data = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Model architecture (same with train)
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
    
    # create model
    model = SimpleNN(model_data['input_size'],
                     model_data['hidden_size'],
                     model_data['output_size'])
    
    # load weights
    model.load_state_dict(model_data['model_state'])
    model.eval() # secure model in eval mode
    
    print(f"\nModel Loaded: {len(model_data['tags'])} intents")
    print(f"Vocalbulary: {len(model_data['all_words'])} words")
    
    return model, model_data


# --- Prediction function ---
def predict_intent(model, sentence, all_Words, tags, device):
    """
    Predict intents form user input
    """
    
    #preprocess
    words = nltk.word_tokenize(sentence)
    word = [lemmatizer.lemmatize(w.lower()) for w in words]
    
    # bag of word
    bag = [1 if word in words else 0 for word in all_Words]
    bag = np.array(bag, dtype=np.float32).reshape(1, -1)
    
    # predict
    with torch.no_grad():
        input_tensor = torch.FloatTensor(bag).to(device)
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        prob, prediction = torch.max(probs, 1)
    return tags[prediction.item()], prob.item()

# --- Get response ---
def get_response(intents_tag, intents_data):
    """
    Get random response for intent
    """
    
    for intent in intents_data:
        if intent['tag'] == intents_tag:
            return random.choice(intent['response'])
    return "Maaf, saya belum paham maksud Anda."


# --- Mian Chat Function ---
def chat():
    """Main Chat Loop"""
    
    print("\n", "-"*50)
    print("Kampus Virtual Assistance")
    print("-"*50)
    print("\nSelamat datang! Saya asisten virtual kampus.")
    print("Saya bisa membantu terkait:")
    print("- Informasi Fakultas dan Jurusan")
    print("- Lokasi Kampus dan Fasilitas")
    print("- Dan Lainnya")
    print("\nKetik 'quit' untuk keluar, 'help' untuk bantuan")
    print("-"*50)
    
    # load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, model_data = load_model()
    
    # simple history
    history = []
    
    while True:
        try:
            # get user input
            user_input = input("\n> ").strip()
            
            # exit command
            if user_input.lower() in ['quit', 'exit', 'keluar', 'bye']:
                print("\nTerima Kasih! Sampai Jumpa Lagi!")
                break
            
            # help command
            if user_input.lower() in ['help', 'bantuan', 'menu']:
                print("\nBantuan:")
                print(f"Pertanyaan yang bisa saya jawab:")
                for intent in model_data['intents']:
                    if intent['patterns']: # show first patterns for example
                        print(f"- {intent['patterns'][0]}")
                print("\nPerintah Khusus:")
                print("- 'quit' - keluar dari chat")
                print("- 'help' - Tampilkan menu bantuan")
                print("- 'history' - Lihat riwayat percakapan")
                continue
            
            # history command
            if user_input.lower() in ['history', 'riwayat']:
                if history:
                    print("\nRiwayat Percakapan:")
                    for i, (user, bot) in enumerate(history[-5:], 1):
                        print(f"{i}. Anda: {user}")
                        print(f"Bot: {bot}")
                else:
                    print("\nBelum ada riwayat percakapn.")
                continue
            
            # skip empty input
            if not user_input:
                continue
            
            # predict intent
            intent, confidence = predict_intent(model,
                                                user_input,
                                                model_data['all_words'],
                                                model_data['tags'],
                                                device)
            
            # get response
            response = get_response(intent, model_data['intents'])
            
            # if confidence low add disclamer
            if confidence < 0.7:
                response = f"{response}\n\n Catatan: Saya tidak terlalu yakin dengan jawaban ini. \nMungkin Anda bisa tanya dengan kata lain?" 
                
            # show response
            print(f"\nBot: {response}")
            
            # Show debug info (optional)
            # print(f"   [Debug] Intent: {intent}, Confidence: {confidence:.1%}")
            
            # save to history
            history.append((user_input, response))
            
        except KeyboardInterrupt:
            print(f"\n\nChat dihentikan. Smapai Jumpa!")
            break
        except Exception as e:
            print(f"\nError {e}")
            print("Coba lagi atau ketik 'help' atau 'bantuan'")

if __name__ == "__main__":
    chat()
            
                

    
