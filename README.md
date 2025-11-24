# 😃 Enhancing Emotional Well-Being in Healthcare  
### *Facial Emotion Classification Using Deep Convolutional Neural Networks (CNNs)*  
#### By: Aditya Girish Anand

## 🔍 Project Overview  
This project builds a **real-time facial emotion recognition system** designed to support emotional well-being monitoring in healthcare environments. Using a **Deep Convolutional Neural Network (CNN)** trained on the **FER-2013 dataset**, the model classifies seven emotions from live video input:  
**Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise**.  


The goal is to enable healthcare providers to detect patient emotions, improve communication, and offer more empathetic, personalized care — especially for children, dementia patients, and individuals unable to express emotions verbally.  


---

## 🧠 Key Features  
- **Real-time webcam emotion detection**  
- **CNN architecture with convolution, pooling, batch norm, dropout, and dense layers**  
- **Softmax-based 7-class emotion classification**  
- **FER-2013 dataset (35,887 images, 48×48 grayscale)**  
- **Haar Cascade face detection for preprocessing**  


---

## 🧪 System Architecture  
The project follows a clear 6-module architecture:  


1. **Input Module** – Captures real-time video frames  
2. **Preprocessing Module** – Face detection (Haar Cascade), region extraction, 48×48 resizing  
3. **CNN Module** – Extracts hierarchical features through convolution & pooling  
4. **Training Process** – Backpropagation with categorical cross-entropy loss  
5. **Output Module** – Predicts emotion via Softmax  
6. **Display Module** – Renders real-time emotion label on screen  

---

## 🎯 Dataset: FER-2013  
According to the dataset table in the report:  


- **Total Images:** 35,887  
- **Image Size:** 48×48 grayscale  
- **Emotion Classes:** Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise  

**Dataset Split**  


| Type | Images | Percentage |
|------|--------|------------|
| Training | 25,210 | 70.24% |
| Testing | 5,984 | 16.67% |
| Validation | 4,683 | 13.04% |

---

## 🛠️ Model Details (CNN Architecture)  
The CNN includes:  
- Four convolutional layers  
- ReLU activation  
- Max pooling  
- Batch normalization  
- Dense layers & dropout  
- Softmax output layer  


**Training Hyperparameters:**  

- Learning rate: 0.0001  
- Decay: 1e-6  
- Batch size: 64  
- Epochs: 15  

---

## 📊 Results  
### ✔ Confusion Matrix & Performance  
The model performs best on:  
- **Happy**  
- **Surprise**  
- **Neutral**  


Struggles with:  
- **Disgust** (low support & recognition rates)

### ✔ Classification Metrics  
Weighted average F1-score ≈ **0.47**  


---

## 💡 Applications in Healthcare  
As described in the paper:  


- Monitoring emotional well-being of patients  
- Understanding children’s emotions (non-verbal communication)  
- Dementia care (detecting distress)  
- Pain assessment through facial expressions  
- Improving doctor–patient communication  

---

## 🛠️ Tech Stack  
- **Python 3**  
- **TensorFlow / Keras**  
- **OpenCV (Haar Cascades)**  
- **NumPy / Pandas**  
- **Matplotlib / Seaborn**


