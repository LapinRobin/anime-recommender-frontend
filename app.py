from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

app = Flask(__name__)
app.secret_key = 'secret'

# Load the data
anime_parquet = pd.read_parquet('static/parquet/anime.parquet')

# Create a TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(anime_parquet['Mod_name'])


# Route to display the initial form, should handle GET to display the form, and optionally POST if submitting to the same endpoint
@app.route('/', methods=['GET', 'POST'])
def index():
    rated_anime = {}
    for key in session.keys():
        print(key, session[key])
        if key.startswith('rating_'):
            mod_name = key.split('_', 1)[1]  # Extract Mod_name from the key
            rated_anime[mod_name] = session[key]

    anime_details = {}
    if rated_anime:
        # Fetch details for each rated anime and include the rating
        anime_details = {
            mod_name: {
                **get_anime_details_by_mod_name(mod_name),  # Unpack the existing details
                'User_rating': rated_anime[mod_name]  # Add the user rating
            }
            for mod_name in rated_anime.keys() if get_anime_details_by_mod_name(mod_name) is not None
        }

        print(anime_details)
        # Identify and remove any entries that did not have details available
        to_remove = [mod_name for mod_name, details in anime_details.items() if not details]
        for mod_name in to_remove:
            anime_details.pop(mod_name)

    return render_template('index.html', rated_anime=anime_details)


def get_anime_details_by_mod_name(mod_name):
    anime = anime_parquet[anime_parquet['Mod_name'] == mod_name]
    if anime.empty:
        return None
    return anime.iloc[0].to_dict()


# Separate route for recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    anime_name = request.form['anime']
    recommendations = get_recommendations(anime_name)
    return render_template('recommendations.html', anime_name=anime_name, recommendations=recommendations)


@app.route('/search', methods=['GET'])
def search():
    Mod_name = request.args.get('Mod_name')

    query = Mod_name
    terms_list = anime_parquet['Mod_name'].tolist()
    processed_query = re.escape(query)
    regex = re.compile(r'\b' + processed_query, re.IGNORECASE)
    suggestions = [anime_parquet[anime_parquet['Mod_name'] == term].to_dict(orient='records')[0] for term in terms_list
                   if regex.search(term)]

    return render_template('search.html', results=suggestions)


@app.route('/description', methods=['GET'])
def description():
    Mod_name = request.args.get('Mod_name')
    anime = anime_parquet[anime_parquet['Mod_name'] == Mod_name]
    if anime.empty:
        anime_data = None
    else:
        anime_data = anime.iloc[0].to_dict()
    return render_template('description.html', anime=anime_data)


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


@app.route('/rate', methods=['POST'])
def rate():
    mod_name = request.form.get('mod_name')  # Retrieve 'mod_name' from the form data
    rating = request.form.get('rating')  # Retrieve the rating
    # Save the rating in the session using 'mod_name'
    session[f'rating_{mod_name}'] = rating
    return jsonify(success=True)


@app.route('/reset_rating', methods=['POST'])
def reset_rating():
    mod_name = request.form.get('mod_name')  # Retrieve 'mod_name' from the form data
    # Remove the rating from the session using 'mod_name'
    if f'rating_{mod_name}' in session:
        del session[f'rating_{mod_name}']
    return jsonify(success=True)


@app.route('/retrieve_rating', methods=['GET'])
def retrieve_rating():
    mod_name = request.args.get('mod_name')
    rating = session.get(f'rating_{mod_name}', None)
    if rating is not None:
        return jsonify(success=True, rating=rating)
    else:
        return jsonify(success=False, message="Rating not found.")


if __name__ == '__main__':
    app.run(debug=True)
