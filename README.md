# 📄 Resume Parser – AI-Powered Resume Analyzer

The **Resume Parser NLP Project** is a smart application designed to extract and analyze critical details from resumes using **Natural Language Processing (NLP)**. Leveraging technologies like **spaCy**, **NLTK**, **PyPDF2**, and **regex**, it transforms unstructured resume data into structured, actionable insights — ideal for ATS (Applicant Tracking Systems) and recruitment automation.


![Screenshot 2025-05-29 094821](https://github.com/user-attachments/assets/b449c3c3-114c-41e6-90c1-4e9e25fca199)
![Screenshot 2025-05-29 095023](https://github.com/user-attachments/assets/abba6b3a-b611-4284-ac66-003bbd64e79a)
![Screenshot 2025-05-29 095031](https://github.com/user-attachments/assets/75fdc48d-5cec-4f9e-866b-069c7cdb7345)






---

## 🚀 Key Features

✅ **Accurate Resume Parsing** – Extracts name, contact info, skills, experience, education, and more  
✅ **Keyword Matching** – Matches candidate skills against job-relevant keywords  
✅ **PDF & DOCX Support** – Supports multiple file types  
✅ **Machine Learning & NLP** – Enhances extraction accuracy using advanced models  
✅ **User-Friendly UI** – Simple web interface for uploading and analyzing resumes  
✅ **ATS Scoring** – Ranks candidates using relevance scoring system  

---

## 🛠 Tech Stack

- **Python 3.8+**
- **spaCy** (with `en_core_web_sm`)
- **NLTK**
- **PyPDF2**
- **python-docx**
- **Pandas**
- **Regex**
- *(Optional UI: Flask / Streamlit)*

---

## 📦 Installation & Setup

### 1️⃣ Install Python Dependencies
```bash
pip install spacy nltk PyPDF2 python-docx pandas
python -m spacy download en_core_web_sm
```

### 2️⃣ Clone or Download the Project
```bash
git clone https://github.com/your-username/resume-parser.git
cd resume-parser
```

### 3️⃣ Prepare Resume Files
Store PDF and DOCX files in a `resumes/` folder (create it if it doesn't exist).

---

## ▶️ Running the Resume Parser

### 🔹 Run via Terminal
```bash
python resume_parser.py
```
This script will parse all resumes in the folder and print or save structured output (CSV/JSON).

---

## 🌐 Optional: Launch Web UI

If your project includes a web interface:

### 🔹 Run Flask App
```bash
python app.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

### 🔹 Or Run Streamlit App
```bash
streamlit run app.py
```

Then open the local URL provided in your terminal.

---

## 📊 Output Format

Parsed data includes:

- **Full Name**
- **Email & Phone**
- **Skills (matched & extracted)**
- **Years of Experience**
- **Education Details**
- **ATS Score**

Results can be saved as CSV/JSON depending on implementation.

---

## 👥 Ideal For

- Recruiters & HR teams  
- Job Portals  
- ATS Developers  
- Resume Screening Tools

---

## 📝 License

MIT License. Feel free to modify and use in commercial or personal projects.

---

Made with ❤️ by Akshay Karthick  
🔗 [LinkedIn](https://www.linkedin.com/in/akshay-karthick-32817a249/) | [GitHub](https://github.com/akshaykarthicks)
