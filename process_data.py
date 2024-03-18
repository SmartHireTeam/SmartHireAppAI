import os
import docx
import fitz  # PyMuPDF for PDF processing
from openai import OpenAI
import numpy as np
import re
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


openai_api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=openai_api_key)


# Function to read text from a .docx file
def read_docx(file_path):
    doc = docx.Document(file_path)
    full_text = [para.text for para in doc.paragraphs]
    return ' '.join(full_text)

# Function to read text from a .pdf file
def read_pdf(file_path):
    with fitz.open(file_path) as doc:
        text = ''
        for page in doc:
            text += page.get_text()
    return text

# General preprocessing of text
def preprocess_text(text):
    # Convert to lower case
    text = str(text).lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize and remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Rejoin tokens
    return ' '.join(tokens)

# Placeholder for your text data
texts = []

def process_data(dir_path):
    # Read and preprocess .docx files
    if dir_path and os.path.exists(dir_path):
        for filename in tqdm(os.listdir(dir_path), desc='Reading JD files'):
            if filename.endswith('.docx'):
                file_path = os.path.join(dir_path, filename)
                text = read_docx(file_path)
                preprocessed_text = preprocess_text(text)
                texts.append(preprocessed_text)

    # Read and preprocess .pdf files
    if dir_path and os.path.exists(dir_path):
        for filename in tqdm(os.listdir(dir_path), desc='Reading Resume files'):
            if filename.endswith('.pdf'):
                file_path = os.path.join(dir_path, filename)
                text = read_pdf(file_path)
                preprocessed_text = preprocess_text(text)
                texts.append(preprocessed_text)


    # Example function to generate embeddings for a list of preprocessed texts
    def generate_embeddings(texts):
        inputs = texts if isinstance(texts, list) else [texts]
        embeddings = []

        for text_chunk in tqdm(inputs, desc='Generating embeddings'):
            response = client.embeddings.create(
                        input=text_chunk,
                        model="text-embedding-3-small" 
                    )
            embeddings.append(response.data[0].embedding)
        # model.encode(texts, show_progress_bar=True)
        return embeddings

    # Assuming `texts` is a list of your preprocessed documents
    embeddings = generate_embeddings(texts)

    # Sentence Transformer model
    # embeddings = model.encode(texts, show_progress_bar=True)

    # Store the embeddings for later use
    return np.save('data_embeddings.npy', embeddings)
    
