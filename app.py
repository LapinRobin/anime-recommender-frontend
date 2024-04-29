from flask import Flask, render_template, request

app = Flask(__name__)

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

if __name__ == '__main__':
    app.run(debug=True)
