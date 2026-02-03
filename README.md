
# 🦠 COVID-19 Test Prediction using ANN

A simple **Artificial Neural Network (ANN)** project to predict whether a person is **COVID-19 Positive or Negative** based on medical data.

This project focuses on **clarity, simplicity, and correctness**, making it suitable for learning and academic use.

---

## ✨ What This Project Does

- Takes patient medical features as input  
- Uses a neural network to learn patterns  
- Predicts COVID-19 test result as:
  - ✅ Positive  
  - ❌ Negative  

---

## 🧠 Model Used

**Artificial Neural Network (ANN)**

### Architecture
- **Input Layer** – Medical features  
- **Hidden Layer**
  - 8 neurons
  - ReLU activation  
- **Output Layer**
  - 1 neuron
  - Sigmoid activation  

> Sigmoid outputs a value between **0 and 1**, which represents the probability of COVID infection.

---

## ⚙️ Tools & Technologies

- Python  
- TensorFlow / Keras  
- NumPy  
- Pandas  
- Scikit-learn  

---

## 📊 Training Details

- **Optimizer:** Adam  
- **Loss Function:** Binary Crossentropy  
- **Metric:** Accuracy  

These are standard choices for **binary medical classification problems**.

---

## 📁 Project Structure

```
├── ann_model.py     # Neural network model
├── train.py         # Model training script
├── dataset.csv      # Medical dataset
├── README.md        # Project documentation
```

---

## 🚀 How to Run

1. Clone the repository
   ```bash
   git clone https://github.com/your-username/covid-ann-prediction.git
   ```

2. Install dependencies
   ```bash
   pip install tensorflow keras numpy pandas scikit-learn
   ```

3. Train the model
   ```bash
   python train.py
   ```

---

## 🔍 How Prediction Works

- Model outputs a probability value  
- If value **≥ 0.5** → **COVID Positive**  
- If value **< 0.5** → **COVID Negative**

---

## ✅ Why This Project Is Useful

- Easy to understand ANN implementation  
- Good example of **binary classification**  
- Suitable for **students and beginners**  
- Low computational cost  

---

## ⚠️ Limitations

- Uses a basic ANN architecture  
- Accuracy depends on data quality  
- Not intended for real-world medical diagnosis  

---

## 🔮 Future Enhancements

- Add more hidden layers  
- Apply Dropout to avoid overfitting  
- Use Precision, Recall, and F1-score  
- Extend to CNN models for X-ray/CT scan images  

---

## 👨‍💻 Author

T.Mourya
B.Tech 3rd Year Student | Deep Learning Enthusiast  

---

## 📜 Disclaimer

This project is **for educational purposes only** and should not be used for real medical diagnosis.
