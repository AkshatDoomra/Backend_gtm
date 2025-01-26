from flask import Flask, request, jsonify
import joblib
import re
import json
import spacy

# Load pre-trained spaCy model
nlp = spacy.load("en_core_web_sm")

# Load the saved model, vectorizer, and label binarizer
model = joblib.load('multi_label_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
mlb = joblib.load('mlb.pkl')

# Load the domain knowledge JSON file
with open('domain_knowledge.json') as f:
    domain_knowledge = json.load(f)

# Function to predict labels (multi-label classification)
def predict_labels(text):
    # Transform the text using the vectorizer
    text_tfidf = vectorizer.transform([text])
    # Predict using the model
    predictions = model.predict(text_tfidf)
    # Transform the prediction into label names
    labels = mlb.inverse_transform(predictions)
    return labels[0]

# Function to extract domain-specific entities from the knowledge base
def extract_from_knowledge_base(text, domain_knowledge):
    entities = []
    for competitor in domain_knowledge['competitors']:
        if re.search(r'\b' + re.escape(competitor) + r'\b', text, flags=re.IGNORECASE):
            entities.append({'competitor': competitor})

    for feature in domain_knowledge['features']:
        if re.search(r'\b' + re.escape(feature) + r'\b', text, flags=re.IGNORECASE):
            entities.append({'feature': feature})

    for keyword in domain_knowledge['pricing_keywords']:
        if re.search(r'\b' + re.escape(keyword) + r'\b', text, flags=re.IGNORECASE):
            entities.append({'pricing_keyword': keyword})

    return entities

# Function to extract entities using spaCy NER
def extract_from_ner(text):
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append({'ner_entity': ent.text, 'ner_label': ent.label_})
    return entities

# Function to generate a summary
def generate_summary(text):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    summary = " ".join(sentences[:2])  # Return the first 1-2 sentences as summary
    return summary

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request body
    try:
        data = request.get_json()  # Parses JSON body from the request
    except Exception as e:
        return jsonify({"error": "Invalid JSON format", "details": str(e)}), 400

    # Validate JSON structure
    if not data or 'text_snippet' not in data:
        return jsonify({"error": "Missing 'text_snippet' in request body"}), 400

    # Extract the text snippet
    text = data['text_snippet']

    # Predict labels using the trained model
    labels = predict_labels(text)

    # Extract domain-specific entities
    knowledge_base_entities = extract_from_knowledge_base(text, domain_knowledge)

    # Extract general entities using NER
    ner_entities = extract_from_ner(text)

    # Generate a summary
    summary = generate_summary(text)

    # Combine entities
    all_entities = knowledge_base_entities + ner_entities

    # Return the results in a JSON response
    response = {
        'predicted_labels': labels,
        'extracted_entities': all_entities,
        'summary': summary
    }

    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
