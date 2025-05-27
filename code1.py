# full_threat_checker.py

# ----------------------------
# Imports
# ----------------------------
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ----------------------------
# Part 1: URL Malicious Classification
# ----------------------------
def url_classification():
    print("\n=== URL Malicious Classification ===")
    # Load URL dataset (first 400 rows)
    df_urls = pd.read_csv('/Users/suryapalsinghbisht/Downloads/malicious_phish.csv').head(400)
    
    # TF-IDF on URL strings (character n-grams)
    vectorizer_url = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5))
    X_url = vectorizer_url.fit_transform(df_urls['url'])
    
    # Encode labels (e.g. "benign" vs "phishing" vs "malware" etc.)
    le_url = LabelEncoder()
    y_url = le_url.fit_transform(df_urls['type'])
    
    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X_url, y_url, test_size=0.2, random_state=42, stratify=y_url
    )
    
    # Train Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Report performance
    print("\nClassification Report on URL Test Set:")
    from sklearn.metrics import classification_report
    print(classification_report(y_test, clf.predict(X_test),
          target_names=le_url.classes_))
    
    # Helper to map any non-benign label to "malicious"
    def map_to_binary(label):
        return 'benign' if label.lower() == 'benign' else 'malicious'
    
    # Predict loop
    while True:
        url = input("\nEnter a URL to classify (or 'exit'): ").strip()
        if url.lower() == 'exit':
            break
        vec = vectorizer_url.transform([url])
        pred = clf.predict(vec)[0]
        human_label = le_url.inverse_transform([pred])[0]
        print(f"→ {map_to_binary(human_label)}")
    print("Exiting URL classification...\n")


# ----------------------------
# Part 2: Email Text Clustering & Threat Detection
# ----------------------------
def email_clustering_and_detection():
    print("=== Email Text Clustering & Threat Detection ===")
    
    # Load emails
    df_emails = pd.read_csv('/Users/suryapalsinghbisht/Downloads/emails.csv')
    texts = df_emails['email_text'].fillna("")
    
    # TF-IDF vectorization
    vectorizer_email = TfidfVectorizer(stop_words='english', max_features=1000)
    X_email = vectorizer_email.fit_transform(texts)
    
    # K-Means into 2 clusters
    kmeans = KMeans(n_clusters=2, random_state=42)
    labels = kmeans.fit_predict(X_email)
    df_emails['cluster'] = labels
    
    # Silhouette score
    sil_score = silhouette_score(X_email, labels)
    print(f"Silhouette Score (2 clusters): {sil_score:.3f}")
    
    # Top keywords per cluster
    def top_keywords(matrix, labels, vect, n=10):
        terms = np.array(vect.get_feature_names_out())
        out = {}
        for c in np.unique(labels):
            center = matrix[labels==c].mean(axis=0)
            idxs = np.argsort(center.A1)[::-1][:n]
            out[c] = list(terms[idxs])
        return out
    
    keywords = top_keywords(X_email, labels, vectorizer_email)
    print("\nTop keywords per cluster:")
    for c, words in keywords.items():
        print(f" Cluster {c}: {words}")
    
    # Heuristic: find which cluster likely contains phishing-style text
    malicious_terms = {"win","prize","click","free","offer","money","password","account"}
    mal_cluster = next((c for c,w in keywords.items() if any(t in w for t in malicious_terms)), 0)
    print(f"\n→ Treating cluster {mal_cluster} as 'malicious-like'")
    
    # Keyword-based threat scan on all emails
    threat_kw = [
        'urgent','alert','suspicious','verify','action required',
        'click here','login','account','password','security'
    ]
    pattern = re.compile(r'\b(' + '|'.join(threat_kw) + r')\b', re.IGNORECASE)
    df_emails['is_threat'] = df_emails['email_text'].apply(lambda t: bool(pattern.search(t)))
    
    print("\nSample detected threat emails:")
    print(df_emails[df_emails['is_threat']].head(5)[['email_text','cluster']])
    
    # Manual check function
    def check_email(text):
        if pattern.search(text):
            return "🚨 POTENTIALLY MALICIOUS"
        else:
            return "✅ SAFE"
    
    # Interactive loop
    while True:
        txt = input("\nEnter email/text to check (or 'exit'): ")
        if txt.lower() == 'exit':
            break
        print("→", check_email(txt))
    print("Exiting email threat checker...")


if __name__ == "__main__":
    url_classification()
    email_clustering_and_detection()
