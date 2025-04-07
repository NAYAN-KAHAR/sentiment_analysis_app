from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

with open('sentiment_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    loaded_vectorizer = pickle.load(vectorizer_file)

def predict_sentiment(comment):
    # Preprocess the new comment using the loaded vectorizer
    comment_tfidf = loaded_vectorizer.transform([comment])

    sentiment_prediction = loaded_model.predict(comment_tfidf)
    
    print(f"Raw sentiment prediction: {sentiment_prediction}")
    
    if sentiment_prediction[0] == 1:
        return 'Neutral'
    elif sentiment_prediction[0] == 2:
        return 'Positive'
    else:
        return 'Negative'
    
   


# Route to display the home page and input form
@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None
    if request.method == "POST":
        new_comment = request.form["comment"]
        if new_comment:
            sentiment = predict_sentiment(new_comment)
    
    return render_template("index.html", sentiment=sentiment)

if __name__ == "__main__":
    app.run(debug=True)
