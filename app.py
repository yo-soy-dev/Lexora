# import streamlit as st
# import pickle
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# # ------------------------------
# # Load saved files
# # ------------------------------
# @st.cache_resource
# def load_resources():
#     model = load_model("lstm_model (1).h5")
#     # model = load_model("lstm_model (1).h5", compile=False)
#     with open("tokenizer.pkl", "rb") as f:
#         tokenizer = pickle.load(f)
#     with open("max_len.pkl", "rb") as f:
#         max_len = pickle.load(f)
#     return model, tokenizer, max_len

# model, tokenizer, max_len = load_resources()

# # ------------------------------
# # Prediction function
# # ------------------------------
# def predict_next_word(text):
#     sequence = tokenizer.texts_to_sequences([text])[0]
#     sequence = pad_sequences([sequence], maxlen=max_len-1, padding='pre')

#     preds = model.predict(sequence, verbose=0)
#     predicted_index = np.argmax(preds)

#     for word, index in tokenizer.word_index.items():
#         if index == predicted_index:
#             return word
#     return ""

# # ------------------------------
# # Streamlit UI
# # ------------------------------
# st.set_page_config(page_title="Next Word Prediction", layout="centered")

# st.title("🧠 Next Word Prediction (LSTM)")
# st.write("Enter a sentence and the model will predict the **next word**.")

# user_input = st.text_input("✍️ Enter text:", placeholder="Type a sentence here...")

# if st.button("Predict Next Word"):
#     if user_input.strip() == "":
#         st.warning("Please enter some text.")
#     else:
#         next_word = predict_next_word(user_input)
#         st.success(f"**Predicted Next Word:** {next_word}")

# # ------------------------------
# # Footer
# # ------------------------------
# st.markdown("---")
# st.caption("LSTM-based Next Word Prediction using Streamlit")






import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------------------
# Load saved files
# ------------------------------
@st.cache_resource
def load_resources():
    //model = load_model("lstm_model.h5")
    model = load_model("lstm_model.h5", compile=False)
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)
    return model, tokenizer, max_len

model, tokenizer, max_len = load_resources()

# ------------------------------
# Prediction functions
# ------------------------------
def predict_next_word(text):
    sequence = tokenizer.texts_to_sequences([text])[0]
    sequence = pad_sequences([sequence], maxlen=max_len-1, padding='pre')
    preds = model.predict(sequence, verbose=0)[0]
    predicted_index = np.argmax(preds)
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    return ""

def predict_top_n(text, n=5):
    sequence = tokenizer.texts_to_sequences([text])[0]
    sequence = pad_sequences([sequence], maxlen=max_len-1, padding='pre')
    preds = model.predict(sequence, verbose=0)[0]
    top_indices = np.argsort(preds)[-n:][::-1]
    index_to_word = {v: k for k, v in tokenizer.word_index.items()}
    results = []
    for idx in top_indices:
        word = index_to_word.get(idx, "")
        if word:
            results.append((word, float(preds[idx])))
    return results

def predict_n_words(text, n=5):
    result = text
    for _ in range(n):
        next_word = predict_next_word(result)
        if not next_word:
            break
        result += " " + next_word
    return result

# ------------------------------
# Page Config & Custom CSS
# ------------------------------
st.set_page_config(page_title="Next Word Prediction", layout="centered", page_icon="🧠")

st.markdown("""
<style>
    body { background-color: #0e1117; }
    .main { background-color: #0e1117; }

    .title-text {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #9ca3af;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #1e1b4b, #1e3a5f);
        border: 1px solid #4f46e5;
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin: 1rem 0;
        text-align: center;
    }
    .prediction-card .label {
        color: #a5b4fc;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .prediction-card .word {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 700;
        margin-top: 0.3rem;
    }
    .sentence-card {
        background: #1f2937;
        border-left: 4px solid #60a5fa;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        color: #e5e7eb;
        font-size: 1.1rem;
        line-height: 1.7;
    }
    .history-item {
        background: #111827;
        border-radius: 10px;
        padding: 0.6rem 1rem;
        margin: 0.3rem 0;
        color: #d1d5db;
        font-size: 0.9rem;
        border: 1px solid #374151;
    }
    .bar-label {
        color: #e5e7eb;
        font-size: 0.95rem;
    }
    .stButton > button {
        background: linear-gradient(90deg, #4f46e5, #0ea5e9);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover {
        opacity: 0.85;
    }
    hr { border-color: #374151; }
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Header
# ------------------------------
st.markdown('<div class="title-text">🧠 Next Word Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">LSTM-powered predictions — type a sentence and see what comes next</div>', unsafe_allow_html=True)

# ------------------------------
# Session State for History
# ------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ------------------------------
# Input
# ------------------------------
user_input = st.text_input("✍️ Enter your text:", placeholder="e.g. The quick brown fox...")

col1, col2 = st.columns(2)
with col1:
    num_words = st.slider("Words to generate", min_value=1, max_value=10, value=1)
with col2:
    top_n = st.slider("Top predictions to show", min_value=3, max_value=10, value=5)

predict_btn = st.button("🔮 Predict")

# ------------------------------
# Prediction Output
# ------------------------------
if predict_btn:
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        # Single next word
        next_word = predict_next_word(user_input)

        st.markdown(f"""
        <div class="prediction-card">
            <div class="label">Next Predicted Word</div>
            <div class="word">✨ {next_word}</div>
        </div>
        """, unsafe_allow_html=True)

        # Multi-word generation
        if num_words > 1:
            full_sentence = predict_n_words(user_input, n=num_words)
            generated_only = full_sentence[len(user_input):]
            display_html = (
                f'<span style="color:#9ca3af">{user_input}</span>'
                f'<span style="color:#60a5fa;font-weight:600">{generated_only}</span>'
            )
            st.markdown(f'<div class="sentence-card">{display_html}</div>', unsafe_allow_html=True)

        # Top-N predictions with probability bars
        st.markdown("#### 📊 Top Predictions")
        top_preds = predict_top_n(user_input, n=top_n)
        for word, prob in top_preds:
            bar_pct = int(prob * 100)
            st.markdown(f'<div class="bar-label">{word} — <b>{prob:.2%}</b></div>', unsafe_allow_html=True)
            st.progress(min(bar_pct, 100))

        # Save to history
        st.session_state.history.insert(0, {
            "input": user_input,
            "predicted": next_word
        })
        st.session_state.history = st.session_state.history[:10]

# ------------------------------
# Prediction History
# ------------------------------
if st.session_state.history:
    st.markdown("---")
    st.markdown("#### 🕘 Prediction History")
    for item in st.session_state.history:
        st.markdown(
            f'<div class="history-item">📝 <b>{item["input"]}</b> → <span style="color:#a78bfa">{item["predicted"]}</span></div>',
            unsafe_allow_html=True
        )

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.caption("Built with ❤️ using LSTM + Streamlit")
