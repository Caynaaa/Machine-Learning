# 🤖 Campus ChatBot - Prototype

## 📋 Deskripsi
**Prototype sederhana AI Chatbot untuk kampus** menggunakan PyTorch dan NLP dasar. Project ini dibuat **hanya untuk tujuan pembelajaran dan memenuhi tugas**, bukan untuk produksi.

## 🎯 Tujuan Pembelajaran
- Memahami dasar Natural Language Processing (NLP)
- Implementasi Neural Network sederhana dengan PyTorch
- Membuat sistem klasifikasi intent untuk chatbot
- Praktik preprocessing teks dengan NLTK
- Pengembangan prototype AI assistant dari nol

## 📁 Struktur Project
```campus_chatbot/
├── data_intents.json               # Dataset training (patterns & responses)
├── train.py                        # Script training model ML
├── chat.py                         # Interface chatbot 
├── simple_chatbot_model.pth        # Model terlatih (hasil training)
└── README.md                       # Documentations
```

## 📊 Dataset
File **data_intents.json** berisi:
```{
  "intents": [
    {
      "tag": "nama_intent",
      "patterns": ["contoh pertanyaan 1", "contoh 2"],
      "responses": ["jawaban 1", "jawaban 2"]
    }
  ]
}
```

Contoh intents yang tersedia:

- **sapaan** - Sapaan pembuka
- **lokasi** - Informasi lokasi kampus
- **jurusan** - Program studi yang tersedia
- **biaya** - Informasi biaya kuliah
- **penutup** - Percakapan penutup

## 🧠 Arsitektur Model
- Input (Bag-of-Words) → Neural Network (3 Layers) → Output (Intent Classification)
- Input: Bag-of-Words dari vocabulary
- Hidden Layers: 2 layer dengan ReLU activation
- Output: Softmax classification ke intent
- Regularization: Dropout 30%



## 🛠️ Teknologi yang Digunakan
- PyTorch - Framework machine learning
- NLTK - Natural Language Processing toolkit
- NumPy - Komputasi numerik
- Python 3.11+ - Bahasa pemrograman

## 📝 Fitur Chatbot
- ✅ Pure ML-based - Tidak ada rule-based logic
- ✅ Multi-intent classification - Bisa mengenali berbagai jenis pertanyaan
- ✅ Confidence scoring - Menampilkan tingkat keyakinan prediksi
- ✅ Interactive interface - Chat interface interaktif
- ✅ Command helpers - Help, history, exit commands
- ✅ Simple context - Riwayat percakapan terbatas

## ⚠️ Batasan (Prototype)
- ❌ Dataset kecil - Hanya contoh terbatas
- ❌ Tidak ada context deep - Percakapan sederhana
- ❌ Bahasa terbatas - Hanya memahami patterns yang dilatih
- ❌ Tidak ada database - Semua data statis di JSON
- ❌ Accuracy terbatas - Hanya untuk demonstrasi


## 📄 Lisensi
Project ini hanya untuk tujuan edukasi dan pembelajaran. Bebas digunakan untuk keperluan akademik dengan menyertakan credit.




