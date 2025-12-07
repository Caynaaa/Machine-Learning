"""
debug_model.py - Debug why model won't load
"""

import os
import sys

print("="*60)
print("🔍 DEBUG MODEL LOADING ISSUE")
print("="*60)

# 1. Check current directory
print("\n📁 1. CURRENT DIRECTORY:")
print(f"   {os.getcwd()}")
print("\n📄 Files in directory:")
files = os.listdir('.')
for f in files:
    print(f"   - {f}")

# 2. Check specific files
print("\n🔍 2. CHECKING REQUIRED FILES:")
required_files = ['simple_chatbot_model.pth', 'data_intents.json', 'chatbot_model.py', 'app.py']
for f in required_files:
    if os.path.exists(f):
        size = os.path.getsize(f)
        print(f"   ✅ {f}: {size:,} bytes")
    else:
        print(f"   ❌ {f}: NOT FOUND!")

# 3. Test importing chatbot_model
print("\n🧪 3. TESTING IMPORT chatbot_model:")
try:
    from chatbot_model import ChatBotModel
    print("   ✅ Successfully imported ChatBotModel")
    
    # Test creating instance
    bot = ChatBotModel()
    print("   ✅ Created ChatBotModel instance")
    
    # Try to load
    print("   🚀 Attempting to load model...")
    try:
        bot.load()
        print("   ✅ SUCCESS: Model loaded!")
        print(f"   - Intents: {len(bot.tags)}")
        print(f"   - Vocabulary: {len(bot.all_words)} words")
        
        # Test prediction
        print("\n   🤖 TEST PREDICTION:")
        intent, confidence = bot.predict("halo")
        print(f"   - 'halo' -> {intent} ({confidence:.2%})")
        
    except Exception as e:
        print(f"   ❌ ERROR loading model: {type(e).__name__}: {e}")
        
except Exception as e:
    print(f"   ❌ ERROR importing: {type(e).__name__}: {e}")

# 4. Check torch installation
print("\n⚙️  4. CHECKING TORCH INSTALLATION:")
try:
    import torch
    print(f"   ✅ torch {torch.__version__}")
    print(f"   ✅ CUDA available: {torch.cuda.is_available()}")
except ImportError:
    print("   ❌ torch not installed!")

# 5. Check file content
print("\n📖 5. CHECKING MODEL FILE STRUCTURE:")
if os.path.exists('simple_chatbot_model.pth'):
    try:
        import torch
        data = torch.load('simple_chatbot_model.pth', map_location='cpu')
        print("   ✅ Model file can be loaded by torch")
        print(f"   Keys in file: {list(data.keys())}")
        
        if 'tags' in data:
            print(f"   Intents in file: {data['tags']}")
        else:
            print("   ❌ 'tags' key missing in model file!")
            
    except Exception as e:
        print(f"   ❌ torch.load() error: {type(e).__name__}: {e}")
else:
    print("   ❌ Model file not found!")

print("\n" + "="*60)
print("🔧 SUGGESTED FIXES:")
print("="*60)

if not os.path.exists('simple_chatbot_model.pth'):
    print("1. ❌ MODEL FILE MISSING!")
    print("   Run: python train.py")
    
if not os.path.exists('data_intents.json'):
    print("2. ❌ DATASET FILE MISSING!")
    print("   Create data_intents.json first")

print("\n💡 Run this command to fix:")
print("   python train.py")

print("="*60)