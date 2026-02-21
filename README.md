# 🧠 NeuroScan AI – Brain Tumor Detection System

A full-stack AI-powered web application that detects brain tumors from MRI scans using a Convolutional Neural Network (CNN).  

This project includes:
- ✅ Deep Learning model (TensorFlow/Keras)
- ✅ Flask REST API backend
- ✅ HTML + Tailwind CSS frontend
- ✅ PDF & Image upload support
- ✅ Production deployment (Render + Vercel)
- ✅ Memory-optimized ML deployment (Lazy Loading + CPU TensorFlow)

---

## 🚀 Live Demo

🌐 Frontend: https://neuro-scan-ai-chi.vercel.app/  
🔗 Backend API: https://neuroscan-ai-backend-es08.onrender.com  

---

## 🧠 Features

- Upload MRI images (`.jpg`, `.jpeg`, `.png`)
- Upload MRI reports in `.pdf` format
- Automatic PDF → Image conversion using PyMuPDF
- CLAHE-based contrast enhancement preprocessing
- CNN-based tumor classification
- Real-time prediction display
- Responsive UI (Mobile + Desktop)
- Contact form with EmailJS integration

---

## 🏗 System Architecture

Frontend (Vercel)  
⬇  
Flask REST API (Render)  
⬇  
TensorFlow CNN Model  

---

## 🛠 Tech Stack

### 🔹 Frontend
- HTML5
- Tailwind CSS
- Vanilla JavaScript
- EmailJS

### 🔹 Backend
- Flask
- Flask-CORS
- Gunicorn
- TensorFlow (CPU version)
- OpenCV
- NumPy
- Scikit-learn
- PyMuPDF

### 🔹 Deployment
- Backend: Render
- Frontend: Vercel
- Version Control: Git & GitHub

---

## ⚙️ Production Optimization

During deployment on Render (Free Tier), the backend initially failed due to:

- Worker timeout errors
- Memory limitations
- TensorFlow startup overhead

To resolve this:

- Replaced `tensorflow` with `tensorflow-cpu`
- Implemented **lazy model loading**
- Removed global model initialization
- Used absolute file paths
- Optimized memory usage

This reduced startup memory spikes and allowed the backend to run successfully on a free-tier server.

---

## 📁 Project Structure

---
NeuroScan-AI/
│
├── frontend/
│ ├── index.html
│ ├── detection.html
│ ├── architecture.html
│ ├── contact.html
│ ├── scripts.js
│ └── assets/
│
├── backend/
│ ├── app.py
│ ├── brain-tumor-model.keras
│ ├── requirements.txt
│ └── runtime.txt
│
└── README.md

---

---

## 📌 Model Details

- CNN Architecture
- Input Size: 128x128
- Binary Classification:
  - 🚨 Tumor
  - ✅ No Tumor
- Preprocessing:
  - CLAHE contrast enhancement
  - Normalization
  - Resize

---

---

## 📜 Previous Version (Streamlit)

Before building this full-stack system, I developed an earlier version using Streamlit:

🔗 GitHub: https://github.com/frhanahmed/Brain-Tumor-Detection.git  

However, due to Streamlit Cloud free-tier limitations:
- Application frequently went to sleep
- Cold start delays
- Unexpected runtime errors

To overcome these limitations and build a more scalable architecture, I redesigned the system using Flask + REST API + Separate frontend deployment.

This version provides:
- Better scalability
- Better deployment control
- Production-level structure
- Improved reliability

---

---

## 👨‍💻 Author

**Farhan Ahmed**    

- LinkedIn: https://www.linkedin.com/in/farhanahmedf21  
- GitHub: https://github.com/frhanahmed  
- Portfolio: https://frhanahmed.github.io/Portfolio/

---

## ⭐ If You Like This Project

Give it a star on GitHub ⭐
