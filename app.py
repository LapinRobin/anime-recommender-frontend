from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

app = Flask(__name__)


# Load the data
anime_parquet = pd.read_parquet('static/parquet/anime.parquet')

# Create a TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(anime_parquet['Mod_name'])

# Route to display the initial form, should handle GET to display the form, and optionally POST if submitting to the same endpoint
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return recommend()  # Call recommend directly if form is posted here
    return render_template('index.html')

# Separate route for recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    anime_name = request.form['anime']
    recommendations = get_recommendations(anime_name)
    return render_template('recommendations.html', anime_name=anime_name, recommendations=recommendations)

def get_recommendations(anime_name):
    # This function should return a list of recommended anime based on the input anime_name
    return ['Naruto', 'One Piece', 'Bleach']

# Simple GET route to display another page
@app.route('/list')
def list():
    return render_template('list.html')



@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    # Getting the search query from some source
    query = request.args.get('term', '')
    
    # List of terms from another source
    terms_list = anime_parquet['Mod_name'].tolist()
    
    # Escaping special characters in the query to avoid regex errors
    processed_query = re.escape(query)
    
    # Creating a regular expression to search for terms starting with the query
    regex = re.compile(r'\b' + processed_query, re.IGNORECASE)
    
    # Filtering the list of terms using the regular expression
    suggestions = [term for term in terms_list if regex.search(term)]
    
    return suggestions






if __name__ == '__main__':
    app.run(debug=True)
