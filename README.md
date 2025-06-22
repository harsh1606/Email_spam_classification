# 📧 Spam Email Detection using LSTM

## 📌 Overview
This project implements a robust spam detection system using **Natural Language Processing (NLP)** techniques and a **Long Short-Term Memory (LSTM)** neural network. It processes and classifies email content as either **ham (legitimate)** or **spam (unsolicited)** with high accuracy.

---

## 🧰 Technologies Used
- Python
- Pandas & NumPy
- BeautifulSoup (HTML parsing)
- Keras with TensorFlow backend
- LSTM for sequential text modeling

---

## 📁 Dataset Structure
- `./Dataset/ham/` — Contains legitimate email files  
- `./Dataset/spam/` — Contains spam email files  
Emails are parsed using the `email` library and converted into plain text using `BeautifulSoup`.

---

## 🔄 Preprocessing Steps
1. Parse `.eml` files to extract text (plain or HTML).
2. Clean the extracted text (remove punctuation, whitespace).
3. Tokenize and convert to sequences using Keras' `Tokenizer`.
4. Pad sequences to a fixed length (250 tokens).

---

## 🧠 Model Architecture
- Embedding Layer: Converts tokens into dense vectors.
- LSTM Layer: Learns contextual patterns in email sequences.
- Dense Layers: For classification into `ham` or `spam`.

```python
model = Sequential()
model.add(Embedding(50000, 64, input_length=250))
model.add(LSTM(64, activation='tanh'))
model.add(Dense(10, activation='relu'))
model.add(Dense(2, activation='softmax'))
```

---

## 🧪 Model Performance
- Trained on 3,000+ emails
- Achieved **99.2% accuracy** on the test set

📊 **Test Accuracy:** 99.2%  
📉 **Test Loss:** 0.038

---

## 🧾 Sample Predictions
```python
new_complaint = ['todays weather is cool ,isnt it']
# Prediction: ham

new_complaint = ['sdhfsdfhykldjfkl h kkjhfsd jkhadk hudfhkjdfhj,h h']
# Prediction: spam
```

---

## ▶️ How to Run
1. Install dependencies:
```bash
pip install pandas numpy beautifulsoup4 tensorflow keras
```
2. Ensure dataset is present in `./Dataset/ham/` and `./Dataset/spam/`.
3. Run the script in your Python environment:
```bash
python spam_detector.py
```

---

## 👤 Author
**Harsh Kumar Tyagi**  
📧 [harshtyagi022@gmail.com](mailto:harshtyagi022@gmail.com)  
🪪 License: MIT
