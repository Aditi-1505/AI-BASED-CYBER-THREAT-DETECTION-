from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from flask_cors import CORS

app = Flask(__name__)

# ----------------------------
# Model Initialization (Load and train once)
# ----------------------------

# --- URL classification setup ---
df_urls = pd.read_csv('/Users/suryapalsinghbisht/Downloads/malicious_phish.csv').head(400)
vectorizer_url = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5))
X_url = vectorizer_url.fit_transform(df_urls['url'])
le_url = LabelEncoder()
y_url = le_url.fit_transform(df_urls['type'])
X_train, X_test, y_train, y_test = train_test_split(
    X_url, y_url, test_size=0.2, random_state=42, stratify=y_url
)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

def map_to_binary(label):
    return 'benign' if label.lower() == 'benign' else 'malicious'

# --- Email clustering & threat detection setup ---
df_emails = pd.read_csv('/Users/suryapalsinghbisht/Downloads/emails.csv')
texts = df_emails['email_text'].fillna("")
vectorizer_email = TfidfVectorizer(stop_words='english', max_features=1000)
X_email = vectorizer_email.fit_transform(texts)

kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(X_email)
df_emails['cluster'] = labels

# Top keywords per cluster function
def top_keywords(matrix, labels, vect, n=10):
    terms = np.array(vect.get_feature_names_out())
    out = {}
    for c in np.unique(labels):
        center = matrix[labels==c].mean(axis=0)
        idxs = np.argsort(center.A1)[::-1][:n]
        out[c] = list(terms[idxs])
    return out

keywords = top_keywords(X_email, labels, vectorizer_email)

malicious_terms = {"win","prize","click","free","offer","money","password","account"}
mal_cluster = next((c for c,w in keywords.items() if any(t in w for t in malicious_terms)), 0)

threat_kw = [
    'urgent','alert','suspicious','verify','action required',
    'click here','login','account','password','security'
]
pattern = re.compile(r'\b(' + '|'.join(threat_kw) + r')\b', re.IGNORECASE)

def check_email_text(text):
    if pattern.search(text):
        return "🚨 POTENTIALLY MALICIOUS"
    else:
        return "✅ SAFE"

# ----------------------------
# Flask routes
# ----------------------------

@app.route('/classify-url', methods=['POST'])
def classify_url():
    data = request.get_json()
    url = data.get('url', '').strip()
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    
    vec = vectorizer_url.transform([url])
    pred = clf.predict(vec)[0]
    human_label = le_url.inverse_transform([pred])[0]
    binary_label = map_to_binary(human_label)
    return jsonify({'url': url, 'classification': binary_label})

@app.route('/check-email', methods=['POST'])
def check_email():
    data = request.get_json()
    email_text = data.get('email_text', '').strip()
    if not email_text:
        return jsonify({'error': 'No email text provided'}), 400
    
    # Check if email text contains threat keywords
    threat_status = check_email_text(email_text)
    
    # Additionally, cluster prediction (optional)
    vec_email = vectorizer_email.transform([email_text])
    cluster_label = kmeans.predict(vec_email)[0]
    cluster_type = 'malicious-like' if cluster_label == mal_cluster else 'benign-like'
    
    return jsonify({
        'email_text': email_text,
        'threat_status': threat_status,
        'cluster_label': int(cluster_label),
        'cluster_type': cluster_type
    })

# Run the app
if __name__ == '__main__':
    CORS(app, resources={r"/*": {"origins": "*"}})
    app.run(debug=True)
