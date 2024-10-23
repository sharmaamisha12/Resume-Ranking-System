from flask import Flask, request, render_template
import os
import re
import PyPDF2
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

# Function to extract text from PDFs
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ''
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:  # Ensure text is not None
            text += page_text
    return text

# Function to extract name and email from resume text
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

# Preprocessing to clean resume text
def clean_resume_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetic characters
    return text

# Function to get BERT embeddings for text
def get_bert_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Function to rank resumes based on embeddings and keyword matching
def rank_resumes_with_combination(job_desc, resumes):
    # Load BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Define important keywords for the job description
    keywords = ['ai', 'ml', 'machine learning', 'nlp', 'python', 'tensorflow', 'pytorch', 'bert', 'scikit-learn']
    negative_keywords = ['marketing', 'seo', 'social media', 'campaign management', 'advertising', 'customer engagement']

    # Get embeddings for job description and resumes
    job_desc_embedding = get_bert_embedding(job_desc, tokenizer, model)
    resume_embeddings = [get_bert_embedding(resume, tokenizer, model) for resume in resumes]

    # Keyword vectorizer
    vectorizer = CountVectorizer(vocabulary=keywords)
    neg_vectorizer = CountVectorizer(vocabulary=negative_keywords)

    job_desc_keywords = vectorizer.transform([job_desc]).toarray()
    resume_keywords = vectorizer.transform(resumes).toarray()

    # Negative keyword vectorization
    resume_neg_keywords = neg_vectorizer.transform(resumes).toarray()

    # Calculate similarity scores
    similarity_scores = []
    for i, resume_embedding in enumerate(resume_embeddings):
        # Cosine similarity for embeddings
        embedding_similarity = np.dot(job_desc_embedding, resume_embedding) / (np.linalg.norm(job_desc_embedding) * np.linalg.norm(resume_embedding))
        
        # Keyword matching score
        keyword_similarity = np.dot(job_desc_keywords[0], resume_keywords[i]) / (np.linalg.norm(job_desc_keywords[0]) * np.linalg.norm(resume_keywords[i]) + 1e-6)
        
        # Negative keyword penalty
        neg_keyword_penalty = np.dot(resume_neg_keywords[i], resume_neg_keywords[i])  # Penalize any negative keywords

        # Adjust scores: heavily penalize irrelevant resumes
        combined_similarity = 0.7 * embedding_similarity + 0.5 * keyword_similarity - 1.0 * neg_keyword_penalty  # Increase penalty

        similarity_scores.append(combined_similarity)
    
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
        
        # Rank resumes with BERT and keyword combination
        similarity_scores = rank_resumes_with_combination(clean_resume_text(job_description), resumes)
        
        for i, score in enumerate(similarity_scores):
            resume_data[i]['similarity'] = round(score, 4)
        
        # Sort resumes by similarity
        results = sorted(resume_data, key=lambda x: x['similarity'], reverse=True)

        return render_template('index.html', results=results)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
