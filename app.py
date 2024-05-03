from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

from subtitles_chatbot import files_embedding, contextualize_q_system_prompt, qa_system_prompt, Document
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

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
        # print(key, session[key])
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

        # Identify and remove any entries that did not have details available
        to_remove = [mod_name for mod_name, details in anime_details.items() if not details]
        for mod_name in to_remove:
            anime_details.pop(mod_name)

        session['recommend_list'] = [(anime['anime_id'], int(anime['User_rating'])) for anime in anime_details.values()]

    return render_template('index.html', rated_anime=anime_details)


def get_anime_details_by_mod_name(mod_name):
    anime = anime_parquet[anime_parquet['Mod_name'] == mod_name]
    if anime.empty:
        return None
    return anime.iloc[0].to_dict()


# Separate route for recommendations
@app.route('/recommendations')
def recommendations():
    recommand_list = session.get('recommand_list', [])
    return render_template('recommendation.html', recommand_list=recommand_list)


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

@app.route('/filter', methods=['GET'])
def filter_anime():
    mod_name = request.args.get('Mod_name', '')
    genre = request.args.get('genre', '')
    type_ = request.args.get('type', '')
    rating = request.args.get('rating', '')

    # Filter the dataset based on the received parameters
    filtered_anime = anime_parquet[
        (anime_parquet['Mod_name'].str.contains(mod_name, case=False, na=False) if mod_name else True) &
        (anime_parquet['Genres'].str.contains(genre, case=False, na=False) if genre else True) &
        (anime_parquet['Type'].str.contains(type_, case=False, na=False) if type_ else True) &
        (anime_parquet['Rating'].str.contains(rating, case=False, na=False) if rating else True)
    ]

    # Convert filtered DataFrame to list of dicts for rendering
    results = filtered_anime.to_dict(orient='records')

    return render_template('results.html', results=results)

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


# Route for the home chatbot, displays another page via an HTML template
@app.route('/home-chatbot')
def home():
    return render_template('home-chatbot.html')


# Route for the general chatbot, displays another page via an HTML template
@app.route('/general-chatbot')
def general_chatbot():
    return render_template('general-chatbot.html')


# Initialization of global variables used for the subtitles chatbot
vectorstore_subtitles = None
retriever_subtitles = None
conversational_rag_chain_subtitles = None
llm_subtitles = None
history_aware_retriever_subtitles = None
session_id = 'user_session'
chat_history_subtitles = None
chat_history_msgs_subtitles = None
store_subtitles = {}


# Function to obtain the chat session history
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store_subtitles:
        store_subtitles[session_id] = ChatMessageHistory()
    return store_subtitles[session_id]


# Route for the subtitles chatbot
@app.route('/subtitles-chatbot')
def subtitles_chatbot():
    return render_template('subtitles-chatbot.html')


# Route to initialize and load the data of the subtitles chatbot
@app.route('/embed_files')
async def embed_files_api():
    global vectorstore_subtitles, retriever_subtitles
    if vectorstore_subtitles is None:
        vectorstore_subtitles = files_embedding()
        try:
            if vectorstore_subtitles is not None:
                retriever_subtitles = vectorstore_subtitles.as_retriever()
                return jsonify({"status": "success"}), 200
            else:
                # Properly handle the case where vectorstore_subtitles is None
                return jsonify({"error": "Failed to create vectorstore_subtitles"}), 500
        except Exception as e:
            # Handle any exception that might occur during files_embedding() processing
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"status": "The embedding is already done"}), 200


# Route to receive and process messages sent to the subtitles chatbot
@app.route('/get_response_subtitles_chatbot', methods=['POST'])
async def get_response_subtitles_chatbot():
    global conversational_rag_chain_subtitles, llm_subtitles, history_aware_retriever_subtitles, session_id, chat_history_subtitles, chat_history_msgs_subtitles, store_subtitles
    user_message = request.json.get('message')

    if not llm_subtitles:
        llm_subtitles = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

    if not history_aware_retriever_subtitles:
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever_subtitles = create_history_aware_retriever(
            llm_subtitles, retriever_subtitles, contextualize_q_prompt
        )

    if not conversational_rag_chain_subtitles:
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm_subtitles, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever_subtitles, question_answer_chain)
        conversational_rag_chain_subtitles = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    chat_history_subtitles = get_session_history(session_id)
    chat_history_msgs_subtitles = await chat_history_subtitles.aget_messages()

    response_msg = conversational_rag_chain_subtitles.invoke(
        {
            "input": user_message,
            "chat_history": chat_history_msgs_subtitles
        },
        config={"configurable": {"session_id": session_id}}
    )

    chat_history_subtitles.add_messages([
        ('human', user_message),
        ('system', response_msg['answer'])
    ])

    return jsonify({"answer": response_msg['answer']})


# Delete chat subtitles history
@app.route('/delete_chat_subtitles_history')
def delete_chat_subtitles_history():
    del store_subtitles[session_id]
    return jsonify({"status": "Chat history deleted"}), 200

if __name__ == '__main__':
    app.run(debug=True, port=8000)
