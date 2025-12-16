"""
SIMPLE Chatbot Training - Optimized for Small Dataset, this model be training with small dataset
"""

# === Import Library ===
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import nltk
from nltk.stem import WordNetLemmatizer
import random

# Set random seed untuk reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Download nltk (FIX TYPO)
def setup_nltk():
    try:
        nltk.data.find('tokenizers/punkt') # -> Tokenizer untuk memecah kalimat menajadi kata-kata
        nltk.data.find('tokenizers/punkt_tab') # -> Dataset tambahan untuk tokenize
        nltk.data.find('corpora/wordnet') # -> Dataset semantic untuk mencari hubungan dan synonim kata
    except LookupError:
        print("Downloading NLTK data...")
        nltk.download('punkt')
        nltk.download('punkt_tab')   
        nltk.download('wordnet')

setup_nltk()

# === Load Dataset ===
with open("./data_intents.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

# === Preprocessing ===
lemmatizer = WordNetLemmatizer()

def preprocess_data(data):
    all_words = []
    tags = []
    xy = []
    
    ignore_symbols = ['?', '!', '.', ',', "'"]
    
    for intent in data["intents"]:
        tag = intent['tag']
        tags.append(tag)
        
        for pattern in intent['patterns']:
            # Tokenize
            words = nltk.word_tokenize(pattern)
            # Filter dan lemmatize
            words = [lemmatizer.lemmatize(w.lower()) 
                    for w in words if w not in ignore_symbols]
            
            all_words.extend(words)
            xy.append((words, tag))
             
    # Remove duplicates and sort
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))
    
    return all_words, tags, xy
            
all_words, tags, xy = preprocess_data(data)

print("=" * 50)
print("DATA STATISTICS")
print("=" * 50)
print(f"Unique words: {len(all_words)}")
print(f"Intents: {len(tags)}")
print(f"Training samples: {len(xy)}")
print(f"Average samples per intent: {len(xy)/len(tags):.1f}")
print("Intents:", tags)
print("=" * 50)

# === Create training data ===
def create_training_data(xy, all_words, tags):
    X = []
    y = []
    
    for (pattern_words, tag) in xy:
        # bag of words
        bag = [1 if word in pattern_words else 0 for word in all_words]
        X.append(bag) 
        y.append(tags.index(tag))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

X, y = create_training_data(xy, all_words, tags)

# === Simple NN Architecture ===
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

# === Dataset class ===
class ChatDataset(Dataset):
    def __init__(self, X, y):
        self.features = torch.FloatTensor(X)
        self.labels = torch.LongTensor(y)
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# === Training setup ===
# Hyperparameters
input_size = len(all_words)
hidden_size = 32  
output_size = len(tags)
batch_size = 8    
learning_rate = 0.001
num_epochs = 300  

dataset = ChatDataset(X, y)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on: {device}")

# Model, Loss, Optimizer
model = SimpleNN(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# === Training Loop ===
print("\n" + "="*50)
print("START TRAINING")
print("="*50)

# Tracking untuk plotting nanti
train_losses = []

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Print progress lebih sering (setiap 10 epoch untuk dataset kecil)
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1:3d}/{num_epochs}] | Loss: {avg_train_loss:.4f}")
    
    # Early stopping sederhana
    if avg_train_loss < 0.01:  # Stop jika loss sangat kecil
        print(f"Early stopping at epoch {epoch+1} (loss < 0.01)")
        break

print("\nTraining Completed!")

# === Evaluate ===
def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

# Evaluate
train_accuracy = evaluate_model(model, train_loader, device)
print(f"Training Accuracy: {train_accuracy:.2f}%")

# === Save Model ===
model_data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags,
    "intents": data["intents"]
}

torch.save(model_data, "simple_chatbot_model.pth")
print(f"\nModel saved as 'simple_chatbot_model.pth'")

# === Test Predictions ===
def predict_intent(model, sentence, all_words, tags, device):
    model.eval()
    
    # Preprocess
    words = nltk.word_tokenize(sentence)
    words = [lemmatizer.lemmatize(w.lower()) for w in words]
    
    # Bag of words
    bag = [1 if word in words else 0 for word in all_words]
    bag = np.array(bag, dtype=np.float32).reshape(1, -1)
    
    # Predict
    with torch.no_grad():
        input_tensor = torch.FloatTensor(bag).to(device)
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        prob, predicted = torch.max(probabilities, 1)
    
    return tags[predicted.item()], prob.item()

print("\n" + "="*50)
print("TESTING MODEL")
print("="*50)

# Test dengan beberapa contoh
test_samples = [
    "halo",
    "dimana kampus",
    "jurusan apa",
    "terima kasih"
]

for sample in test_samples:
    intent, confidence = predict_intent(model, sample, all_words, tags, device)
    print(f"'{sample}'-> {intent} ({confidence:.1%})")

# === Test Predictions dengan RESPONSE LENGKAP ===
def get_response(intent_tag, intents_data):
    """Get full response for an intent"""
    for intent in intents_data:
        if intent['tag'] == intent_tag:
            return random.choice(intent['response'])
    return "Maaf, saya tidak mengerti."

print("\n" + "="*50)
print("TESTING MODEL - DENGAN RESPONSE LENGKAP")
print("="*50)

# Test dengan beberapa contoh
test_samples = [
    "halo",
    "dimana kampus",
    "jurusan apa",
    "biaya kuliah berapa",
    "terima kasih"
]

print("\nTesting dengan response lengkap:")
print("-" * 50)

for sample in test_samples:
    intent, confidence = predict_intent(model, sample, all_words, tags, device)
    response = get_response(intent, data["intents"])
    
    print(f"\nUser: '{sample}'")
    print(f"\nIntent: {intent} ({confidence:.1%})")
    print(f"\nResponse: {response}")

print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print("Model siap digunakan untuk chat interface!")
print("Jalankan: python chat.py")