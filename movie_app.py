import streamlit as st
import joblib
import numpy as np

model = joblib.load('movie_model.pkl')

genre_mapping = {0: 'Action', 1: 'Adventure', 2: 'Animation', 3: 'Comedy', 4: 'Crime',
                 5: 'Documentary', 6: 'Drama', 7: 'Family', 8: 'Fantasy', 9: 'History',
                 10: 'Horror', 11: 'Music', 12: 'Mystery', 13: 'Romance', 14: 'Science Fiction',
                 15: 'Thriller', 16: 'War', 17: 'Western'}
genre_reverse = {v: k for k, v in genre_mapping.items()}

st.title('Movie Profitability Predictor')
st.caption('Predicts whether a movie will make back its budget based on pre-release factors.')

budget = st.number_input('Budget ($)', min_value=0, value=50000000, step=1000000)
runtime = st.number_input('Runtime (minutes)', min_value=0, value=110, step=1)
release_month = st.selectbox('Release Month', list(range(1, 13)), 
                              format_func=lambda x: ['Jan','Feb','Mar','Apr','May','Jun',
                                                      'Jul','Aug','Sep','Oct','Nov','Dec'][x-1])
is_english = st.radio('Language', ['English', 'Non-English']) == 'English'
genre = st.selectbox('Primary Genre', list(genre_reverse.keys()))

if st.button('Predict'):
    features = np.array([[budget, runtime, release_month, int(is_english), genre_reverse[genre]]])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    if prediction == 1:
        st.success(f'Likely PROFITABLE — {probability[1]*100:.1f}% confidence')
    else:
        st.error(f'Likely UNPROFITABLE — {probability[0]*100:.1f}% confidence')
