# 🧠 Lexora — Next Word Prediction using LSTM

## 📌 Overview

**Lexora** is a deep learning-based **Next Word Prediction System** built using **LSTM (Long Short-Term Memory)** and an interactive **Streamlit UI**.
It predicts the most probable next word in a sentence and can intelligently generate multi-word continuations.

---

## 🚀 Features

* ✨ Smart **Next Word Prediction**
* 📊 Displays **Top-N predictions with probabilities**
* 📝 Generates **multi-word sentence completions**
* 🕘 Maintains **prediction history**
* 🎨 Clean and responsive **Streamlit UI**
* ⚡ Fast inference using trained LSTM model

---

## 🏗️ Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python
* **Deep Learning:** TensorFlow / Keras
* **Model:** LSTM (Long Short-Term Memory)

### 📚 Libraries Used:

* NumPy
* TensorFlow
* Pickle
* Streamlit

---

## 🧠 Model Architecture

```id="arch1"
Input Text → Tokenizer → Sequence → Padding → Embedding → LSTM → Dense → Softmax → Predicted Word
```

---

## 📂 Project Structure

```id="struct2"
lexora/
│
├── app.py
├── lstm_model.h5
├── tokenizer.pkl
├── max_len.pkl
├── requirements.txt
├── runtime.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash id="clone1"
git clone https://github.com/your-username/lexora.git
cd lexora
```

### 2️⃣ Install Dependencies

```bash id="install1"
pip install -r requirements.txt
```

### 3️⃣ Run the App

```bash id="run1"
streamlit run app.py
```

---

## 📊 Example

| Input Text          | Output  |
| ------------------- | ------- |
| I love to eat       | pizza   |
| Machine learning is | amazing |
| The quick brown     | fox     |

---

## 💡 How It Works

1. User enters text
2. Tokenizer converts text into sequences
3. Padding ensures fixed input length
4. LSTM predicts next word probabilities
5. Top predictions are displayed
6. Optional multi-word generation extends the sentence

---

## 🔮 Future Improvements

* Add Transformer-based models (GPT-like)
* Train on larger datasets
* Add grammar correction
* Deploy on cloud
* Add speech-to-text
* Multi-language support

---

## 👨‍💻 Author

**Devansh Kumar Tiwari**

---

## 📜 License

MIT License

---

## ⭐ Support

If you like this project, consider giving it a ⭐ on GitHub!
