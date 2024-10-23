Folder Structure on F: Drive
makefile
Copy code
F:/
│
└─── AI Minor Project
     │   app.py
     │
     └─── templates
          │   index.html
1. Folder: AI Minor Project
This is the main project folder where all your files and subfolders are organized.

app.py: The Python file containing the Flask backend.
templates/: A folder containing the HTML templates for your Flask app.
2. File: app.py
This file contains the Flask application that handles the logic and routing for your Resume Ranker. Here's the sample code that goes into app.py:

python
Copy code
from flask import Flask, request, render_template
import PyPDF2
import re
from transformers import BertTokenizer, BertModel
import numpy as np

app = Flask(__name__)

# Function to extract text from PDF
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to clean the resume text
def clean_resume_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# Function to extract name and email from resume
def extract_entities(text):
    email = re.findall(r'\S+@\S+', text)
    email = email[0] if email else "Unknown Email"
    
    lines = text.split("\n")
    name = "Unknown Name"
    for line in lines[:10]:
        if re.match(r"(name|address|contact|email|phone)", line, re.IGNORECASE):
            continue
        words = line.strip().split()
        if len(words) >= 2:
            name = " ".join(words[:2])
            break
    return name.strip(), email

# Function to generate BERT embeddings
def get_bert_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Function to rank resumes
def rank_resumes_with_combination(job_desc, resumes):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    keywords = ['ai', 'ml', 'machine learning', 'nlp', 'python', 'tensorflow', 'pytorch']
    job_desc_embedding = get_bert_embedding(job_desc, tokenizer, model)
    resume_embeddings = [get_bert_embedding(resume, tokenizer, model) for resume in resumes]

    similarity_scores = []
    for i, resume_embedding in enumerate(resume_embeddings):
        embedding_similarity = np.dot(job_desc_embedding, resume_embedding) / (np.linalg.norm(job_desc_embedding) * np.linalg.norm(resume_embedding))
        similarity_scores.append(embedding_similarity)
    
    return similarity_scores

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        job_description = request.form['job_description']
        resume_files = request.files.getlist('resume_files')
        
        resumes = []
        resume_data = []
        
        for resume_file in resume_files:
            resume_text = extract_text_from_pdf(resume_file)
            name, email = extract_entities(resume_text)
            resumes.append(clean_resume_text(resume_text))
            resume_data.append({'name': name, 'email': email, 'text': resume_text})
        
        similarity_scores = rank_resumes_with_combination(clean_resume_text(job_description), resumes)
        
        for i, score in enumerate(similarity_scores):
            resume_data[i]['similarity'] = round(score, 4)
        
        results = sorted(resume_data, key=lambda x: x['similarity'], reverse=True)
        return render_template('index.html', results=results)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
3. Folder: templates
This folder holds your HTML files. In this case, you have one file, index.html.

4. File: templates/index.html
This is your HTML file for the front-end interface.

html
Copy code
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Resume Ranker</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            padding: 20px;
        }
        h1 {
            color: #333;
        }
        form {
            margin-bottom: 20px;
        }
        textarea, input[type="file"] {
            width: 100%;
            padding: 10px;
            margin: 10px 0;
        }
        input[type="submit"] {
            padding: 10px 20px;
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        table, th, td {
            border: 1px solid black;
        }
        th, td {
            padding: 10px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
    </style>
</head>
<body>

<h1>Resume Ranker</h1>

<form action="/" method="post" enctype="multipart/form-data">
    <label for="job_description">Job Description:</label>
    <textarea name="job_description" id="job_description" rows="5" required></textarea>
    
    <label for="resume_files">Upload Resumes (PDF):</label>
    <input type="file" name="resume_files" id="resume_files" accept=".pdf" multiple required>
    
    <input type="submit" value="Rank Resumes">
</form>

{% if results %}
<h2>Ranked Resumes</h2>
<table>
    <thead>
        <tr>
            <th>Rank</th>
            <th>Name</th>
            <th>Email</th>
            <th>Score</th>
        </tr>
    </thead>
    <tbody>
        {% for resume in results %}
        <tr>
            <td>{{ loop.index }}</td>
            <td>{{ resume.name }}</td>
            <td>{{ resume.email }}</td>
            <td>{{ resume.similarity }}</td>
        </tr>
        {% endfor %}
    </tbody>
</table>
{% endif %}

</body>
</html>
Running the Project
To run the project:

Open a terminal and navigate to F:/AI Minor Project.
Install the required packages:
bash
Copy code
pip install flask transformers torch PyPDF2 scikit-learn
Run the Flask app:
bash
Copy code
python app.py
Go to http://127.0.0.1:5000/ in your browser to see the web interface.







