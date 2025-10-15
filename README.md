
# NHL Hockey Outcome Predictor (Streamlit App)

This is a Streamlit web application that predicts whether an NHL team will win or lose based on input statistics like Goals, Shots, Fenwick For, Corsi For, and other advanced metrics. The model uses a trained Random Forest Classifier to generate real-time predictions.

##  Features
- Predict game outcome (Win or Lose)
- Interactive user inputs for various game stats
- Real-time result display with `st.metric()`
- Clean Streamlit UI
- Responsive app

## Input Stats Used
- Goals
- Shots
- Fenwick For
- Corsi For
- FF% (Fenwick For %)
- SF% (Shots For %)

## Machine Learning Model
- Algorithm: Random Forest Classifier
- Libraries: `scikit-learn`, `pandas`, `streamlit`, `joblib`


# Tech Stack
- Frontend: Streamlit
- Backend: FastAPI (Optional if separated)
- ML Tools: Scikit-learn, Pandas
-Deployment: Streamlit Cloud

# How to Run Locally

1. Clone the repo:
    ```bash
    git clone https://github.com/your-username/hockey-predictor-app.git
    cd hockey-predictor-app
    ```

2. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the app:
    ```bash
    streamlit run app.py
    ```

---

# Deploy to Streamlit Cloud

1. Go to [https://share.streamlit.io](https://share.streamlit.io)
2. Click New App
3. Connect your GitHub repo
4. Select the branch and set `app.py` as the main file
5. Hit Deploy

---


# Game-Predictor NHL
NHL hockey game predictor that predicts whether the team will win or lose based on data. Built with Streamlit.

