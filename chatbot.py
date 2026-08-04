import os
import json
import random

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.dirname(__file__)


@st.cache_resource
def load_model():
    """Load intents and train the classifier once per app session."""
    filename = next(
        (
            os.path.join(BASE_DIR, path)
            for path in ('intents.json', 'intent.json')
            if os.path.exists(os.path.join(BASE_DIR, path))
        ),
        None,
    )
    if filename is None:
        raise FileNotFoundError('No intents.json or intent.json file found in the app folder.')

    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read().strip()
    if not text:
        raise ValueError(f'{filename} is empty')

    intents = json.loads(text)

    patterns = []
    tags = []
    for intent in intents:
        for pattern_text in intent.get('patterns', []):
            tags.append(intent.get('tag', ''))
            patterns.append(pattern_text)

    if not patterns:
        raise ValueError('No training patterns found in intents file.')

    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(patterns)

    clf = LogisticRegression(random_state=0, max_iter=10000)
    clf.fit(x, tags)

    return intents, vectorizer, clf


def get_response(input_text: str, intents, vectorizer, clf) -> str:
    input_vector = vectorizer.transform([input_text])
    tag = clf.predict(input_vector)[0]
    for intent in intents:
        if intent.get('tag') == tag:
            return random.choice(intent.get('responses', ['Sorry, I do not understand.']))
    return 'Sorry, I do not understand.'


def main():
    st.set_page_config(page_title='Chatbot', page_icon='🤖')
    st.title('Sports Chatbot')
    st.write('Ask me about football, cricket, basketball, tennis, badminton, or volleyball.')

    intents, vectorizer, clf = load_model()

    if 'history' not in st.session_state:
        st.session_state.history = []

    with st.form(key='chat_form', clear_on_submit=True):
        user_input = st.text_input('You:')
        submitted = st.form_submit_button('Send')

    if submitted and user_input:
        response = get_response(user_input, intents, vectorizer, clf)
        st.session_state.history.append(('You', user_input))
        st.session_state.history.append(('Bot', response))

    for speaker, message in st.session_state.history:
        st.write(f'**{speaker}:** {message}')


if __name__ == '__main__':
    main()
