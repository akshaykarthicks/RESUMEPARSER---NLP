from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from pdfminer.high_level import extract_text
import spacy
import re
import os
from werkzeug.utils import secure_filename
from collections import Counter
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  # Define the directory to store uploaded files
app.secret_key = 'your_secret_key'  # Required for flash messages

# Define allowed extensions
ALLOWED_EXTENSIONS = {'pdf'}

# Regular expressions for phone number and email extraction
PHONE_REG = re.compile(r'[\+$$]?[1-9][0-9 .\-\($$]{8,}[0-9]')
EMAIL_REG = re.compile(r'[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+')

# Load the English NLP model
nlp = spacy.load("en_core_web_sm")

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    try:
        return extract_text(pdf_path)
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {str(e)}")
        return ""

# Function to extract phone numbers
def extract_phone_number(resume_text):
    phone = re.findall(PHONE_REG, resume_text)
    if phone:
        number = ''.join(phone[0])
        if resume_text.find(number) >= 0 and len(number) < 16:
            return number
    return None

# Function to extract email addresses
def extract_emails(resume_text):
    return re.findall(EMAIL_REG, resume_text)

# Function to extract names using proper noun detection
def extract_name(resume_text):
    doc = nlp(resume_text)
    person = [token.text for token in doc if token.pos_ == 'PROPN']
    if len(person) >= 2:
        return ' '.join(person[:2])
    return None

# Function to extract skills based on keywords
def extract_skills(resume_text):
    doc = nlp(resume_text)
    skills = []
    skill_keywords = ['python', 'flask', 'nodejs', 'html', 'css', 'js', 'c/c++', 'java', 
                      'machine learning', 'tamil', 'data analysis', 'communication', 
                      'project management', 'teamwork', 'problem solving', 'negotiation skills', 
                      'opencv', 'web developer']
    for token in doc:
        if token.text.lower() in skill_keywords:
            skills.append(token.text)
    return list(set(skills))  # Return unique skills

# Function to search keyword in all uploaded PDFs
def search_keyword_in_pdfs(directory, query):
    result = []
    if not os.path.exists(directory):
        return result
    
    query = query.lower()  # Convert query to lowercase for case-insensitive matching
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory, filename)
            txt = extract_text_from_pdf(pdf_path)
            if query in txt.lower():
                name = extract_name(txt)
                if not name:
                    name = filename  # Use filename as fallback if no name is extracted
                result.append(name)
    return list(set(result))  # Return unique names
 # Return unique names

# Add new function for detailed analysis
def perform_detailed_analysis(text):
    analysis = {
        'education': extract_education(text),
        'experience': extract_experience(text),
        'certifications': extract_certifications(text),
        'languages': extract_languages(text),
        'technical_skills': extract_technical_skills(text),
        'soft_skills': extract_soft_skills(text),
        'projects': extract_projects(text),
        'ats_analysis': calculate_ats_score(text),
        'word_cloud_data': generate_word_cloud_data(text),
        'skill_distribution': calculate_skill_distribution(text),
        # Add summary statistics
        'summary_stats': {
            'total_experience_years': calculate_total_experience(text),
            'education_level': determine_education_level(text),
            'skill_match_score': calculate_skill_match_score(text),
            'profile_completeness': calculate_profile_completeness(text)
        }
    }
    return analysis

def extract_education(text):
    education_keywords = ['Bachelor', 'Master', 'PhD', 'B.Tech', 'M.Tech', 'B.E.', 'M.E.', 'BSc', 'MSc']
    doc = nlp(text)
    education = []
    
    # Split text into lines for better analysis
    lines = text.split('\n')
    for line in lines:
        if any(keyword in line for keyword in education_keywords):
            education.append(line.strip())
    return education

def extract_experience(text):
    # Look for common job title keywords and company names
    experience_keywords = ['experience', 'work', 'employment', 'job', 'position']
    lines = text.split('\n')
    experience = []
    is_experience_section = False
    
    for line in lines:
        line = line.strip()
        if any(keyword.lower() in line.lower() for keyword in experience_keywords):
            is_experience_section = True
            experience.append(line)
        elif is_experience_section and line:
            experience.append(line)
        elif is_experience_section and not line:
            is_experience_section = False
    return experience

def extract_certifications(text):
    cert_keywords = ['certification', 'certificate', 'certified', 'diploma']
    lines = text.split('\n')
    certifications = []
    
    for line in lines:
        if any(keyword.lower() in line.lower() for keyword in cert_keywords):
            certifications.append(line.strip())
    return certifications

def extract_languages(text):
    common_languages = ['English', 'Spanish', 'French', 'German', 'Chinese', 'Japanese', 
                       'Hindi', 'Arabic', 'Russian', 'Portuguese', 'Italian']
    found_languages = []
    
    for language in common_languages:
        if language.lower() in text.lower():
            found_languages.append(language)
    return found_languages

def extract_technical_skills(text):
    technical_keywords = [
        'python', 'java', 'javascript', 'html', 'css', 'sql', 'react', 'angular',
        'node.js', 'docker', 'kubernetes', 'aws', 'azure', 'git', 'linux',
        'machine learning', 'data analysis', 'artificial intelligence',
        'devops', 'cloud computing', 'database', 'api', 'rest', 'graphql'
    ]
    found_skills = []
    
    for skill in technical_keywords:
        if skill.lower() in text.lower():
            found_skills.append(skill)
    return found_skills

def extract_soft_skills(text):
    soft_skills_keywords = [
        'leadership', 'communication', 'teamwork', 'problem solving',
        'time management', 'critical thinking', 'adaptability', 'creativity',
        'collaboration', 'organization', 'presentation', 'negotiation'
    ]
    found_skills = []
    
    for skill in soft_skills_keywords:
        if skill.lower() in text.lower():
            found_skills.append(skill)
    return found_skills

def extract_projects(text):
    project_keywords = ['project', 'developed', 'implemented', 'created', 'built']
    lines = text.split('\n')
    projects = []
    is_project_section = False
    
    for line in lines:
        line = line.strip()
        if any(keyword.lower() in line.lower() for keyword in project_keywords):
            is_project_section = True
            projects.append(line)
        elif is_project_section and line:
            projects.append(line)
        elif is_project_section and not line:
            is_project_section = False
    return projects

# Add ATS scoring function
def calculate_ats_score(text, job_requirements=None):
    if job_requirements is None:
        # Default job requirements - can be customized
        job_requirements = {
            'education': 10,
            'experience': 25,
            'technical_skills': 30,
            'soft_skills': 15,
            'certifications': 10,
            'formatting': 10
        }
    
    score = 0
    feedback = []
    
    # Education score
    education = extract_education(text)
    if education:
        score += job_requirements['education']
        feedback.append({"category": "Education", "status": "✅ Found relevant education details"})
    else:
        feedback.append({"category": "Education", "status": "❌ Education details not clearly mentioned"})
    
    # Experience score
    experience = extract_experience(text)
    exp_score = min(len(experience) * 5, job_requirements['experience'])
    score += exp_score
    if experience:
        feedback.append({"category": "Experience", "status": f"✅ Found {len(experience)} relevant experience entries"})
    else:
        feedback.append({"category": "Experience", "status": "❌ Work experience not clearly structured"})
    
    # Technical skills score
    tech_skills = extract_technical_skills(text)
    tech_score = min(len(tech_skills) * 3, job_requirements['technical_skills'])
    score += tech_score
    if tech_skills:
        feedback.append({"category": "Technical Skills", "status": f"✅ Found {len(tech_skills)} relevant technical skills"})
    else:
        feedback.append({"category": "Technical Skills", "status": "❌ Technical skills not clearly listed"})
    
    # Soft skills score
    soft_skills = extract_soft_skills(text)
    soft_score = min(len(soft_skills) * 3, job_requirements['soft_skills'])
    score += soft_score
    if soft_skills:
        feedback.append({"category": "Soft Skills", "status": f"✅ Found {len(soft_skills)} relevant soft skills"})
    else:
        feedback.append({"category": "Soft Skills", "status": "❌ Soft skills not clearly mentioned"})
    
    # Certifications score
    certifications = extract_certifications(text)
    cert_score = min(len(certifications) * 5, job_requirements['certifications'])
    score += cert_score
    if certifications:
        feedback.append({"category": "Certifications", "status": f"✅ Found {len(certifications)} certifications"})
    else:
        feedback.append({"category": "Certifications", "status": "❌ No certifications found"})
    
    # Formatting score (basic checks)
    format_score = job_requirements['formatting']
    if len(text.split('\n')) > 10:  # Has proper line breaks
        format_score *= 0.5
        if re.search(r'^[A-Z]', text, re.MULTILINE):  # Proper capitalization
            format_score *= 1
        else:
            format_score *= 0.5
    score += format_score
    
    return {
        "total_score": round(score, 2),
        "max_score": sum(job_requirements.values()),
        "feedback": feedback,
        "section_scores": {
            "Education": round(job_requirements['education'] if education else 0, 2),
            "Experience": round(exp_score, 2),
            "Technical Skills": round(tech_score, 2),
            "Soft Skills": round(soft_score, 2),
            "Certifications": round(cert_score, 2),
            "Formatting": round(format_score, 2)
        }
    }

def generate_word_cloud_data(text):
    # Generate word frequency data for word cloud
    words = re.findall(r'\b\w+\b', text.lower())
    word_freq = Counter(words)
    # Filter out common words and return top 50 words
    common_words = set(['the', 'and', 'or', 'in', 'at', 'to', 'for', 'of', 'with'])
    return [{
        'text': word,
        'value': count
    } for word, count in word_freq.most_common(50) if word not in common_words]

def calculate_skill_distribution(text):
    tech_skills = extract_technical_skills(text)
    soft_skills = extract_soft_skills(text)
    certifications = extract_certifications(text)
    
    return {
        'labels': ['Technical Skills', 'Soft Skills', 'Certifications'],
        'data': [len(tech_skills), len(soft_skills), len(certifications)]
    }

def calculate_total_experience(text):
    # Extract years from experience sections
    experience = extract_experience(text)
    total_years = 0
    year_pattern = re.compile(r'(\d+)[\s-]*year')
    
    for exp in experience:
        matches = year_pattern.findall(exp.lower())
        total_years += sum(int(year) for year in matches)
    
    return total_years

def determine_education_level(text):
    education = extract_education(text)
    levels = {
        'PhD': 4,
        'Master': 3,
        'Bachelor': 2,
        'Diploma': 1
    }
    
    highest_level = 0
    for edu in education:
        for level, score in levels.items():
            if level.lower() in edu.lower():
                highest_level = max(highest_level, score)
    
    return {
        0: 'Not Specified',
        1: 'Diploma Level',
        2: 'Bachelor Level',
        3: 'Master Level',
        4: 'Doctorate Level'
    }.get(highest_level, 'Not Specified')

def calculate_skill_match_score(text):
    tech_skills = extract_technical_skills(text)
    soft_skills = extract_soft_skills(text)
    
    # Weight technical skills more heavily
    return min(100, (len(tech_skills) * 8 + len(soft_skills) * 4))

def calculate_profile_completeness(text):
    sections = {
        'education': bool(extract_education(text)),
        'experience': bool(extract_experience(text)),
        'technical_skills': bool(extract_technical_skills(text)),
        'soft_skills': bool(extract_soft_skills(text)),
        'certifications': bool(extract_certifications(text)),
        'projects': bool(extract_projects(text)),
        'contact': bool(extract_emails(text) or extract_phone_number(text))
    }
    
    completed_sections = sum(sections.values())
    total_sections = len(sections)
    
    return round((completed_sections / total_sections) * 100)

# Route to handle uploads and analysis
@app.route('/', methods=['GET', 'POST'])
def index():
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])  # Create the uploads directory if it doesn't exist

    results = []
    result_from_chat = []
    uploaded_pdfs = []

    if request.method == 'POST':
        if 'pdf_files' not in request.files:
            flash('No file part')
            return redirect(request.url)

        uploaded_files = request.files.getlist("pdf_files")  # Get list of uploaded files

        for pdf_file in uploaded_files:
            if pdf_file and allowed_file(pdf_file.filename):
                filename = secure_filename(pdf_file.filename)
                pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                pdf_file.save(pdf_path)

                txt = extract_text_from_pdf(pdf_path)

                name = extract_name(txt)
                emails = extract_emails(txt)
                phone_number = extract_phone_number(txt)
                skills = extract_skills(txt)

                results.append({
                    'name': name,
                    'emails': emails,
                    'phone_number': phone_number,
                    'skills': skills,
                    'filename': filename
                })

        # Keyword search
        query = request.form.get('query')
        if query:
            result_from_chat = search_keyword_in_pdfs(app.config['UPLOAD_FOLDER'], query)

        # Get list of uploaded PDF filenames
        uploaded_pdfs = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.endswith('.pdf')]

    return render_template('index.html', results=results, result_from_chat=result_from_chat, uploaded_pdfs=uploaded_pdfs)

# Route to serve uploaded files for preview
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except FileNotFoundError:
        return "File not found", 404

# Route to clear all uploaded files
@app.route('/clear_uploads', methods=['POST'])
def clear_uploads():
    # Remove all files from the uploads directory
    for file_name in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Redirect back to the index page after clearing uploads
    return redirect(url_for('index'))

# Modify the index route to handle detailed analysis
@app.route('/detailed_analysis/<filename>')
def detailed_analysis(filename):
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        text = extract_text_from_pdf(file_path)
        analysis = perform_detailed_analysis(text)
        return analysis
    except Exception as e:
        return {"error": str(e)}, 400

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
