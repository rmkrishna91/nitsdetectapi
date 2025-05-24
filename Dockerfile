FROM python:3.12-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# Download required NLTK data
RUN python -m nltk.downloader punkt punkt_tab averaged_perceptron_tagger_eng stopwords tagsets 
 


# Download spaCy model
RUN python -m spacy download en_core_web_sm

EXPOSE 80

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
