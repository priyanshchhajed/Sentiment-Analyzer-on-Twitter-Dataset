# sentiment_app/views.py
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from nltk.sentiment import SentimentIntensityAnalyzer
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import os
import json
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import nltk
from nltk.corpus import stopwords
from django.core.files.storage import default_storage
from django.http import HttpResponse
from django.core.paginator import Paginator

nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

# Load model and vectorizer from the files
model_path = os.path.join('sentiment_app', 'ml_models', 'sentiment_svm_model.pkl')
vectorizer_path = os.path.join('sentiment_app', 'ml_models', 'tfidf_vectorizer.pkl')
scaler_path = os.path.join('sentiment_app', 'ml_models', 'scaler.pkl')

svm_model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)
scaler = joblib.load(scaler_path)

# Initialize SentimentIntensityAnalyzer (VADER)
sia = SentimentIntensityAnalyzer()

categories = {
    "Technology": ["tech", "software", "AI", "data", "computer", "cloud", "robotics", "internet"],
    "Entertainment": ["movie", "music", "show", "film", "concert", "actor", "director"],
    "Politics": ["election", "government", "president", "policy", "vote", "law", "senate"],
    "Sports": ["football", "basketball", "cricket", "match", "goal", "tennis", "athlete"],
    "General Fun": ["fun", "joke", "meme", "party", "game", "laugh", "entertainment"],
    "News": ["breaking", "news", "report", "update", "headline", "journalist"]
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\@\w+|\#", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

def classify_category(text):
    for category, keywords in categories.items():
        if any(word in text for word in keywords):
            return category
    return "Other"

def analyze_sentiment(text):
    try:
        if not text:
            return None, None, "No text to analyze."

        sentence_polarity = sia.polarity_scores(text)["compound"]

        if sentence_polarity >= 0.05:
            sentiment = "Positive"
        elif sentence_polarity <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        return sentiment, sentence_polarity, None

    except Exception as e:
        print("79", str(e))
        return None, None, str(e)

def get_image_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return encoded

def home(request):
    return render(request, 'sentiment_app/home.html')

@csrf_exempt
def analyze_text(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            text = data.get("text")

            if not text:
                return JsonResponse({"error": "No text provided."}, status=400)

            sentiment, polarity, error = analyze_sentiment(text)

            if error:
                return JsonResponse({"error": error}, status=500)

            return JsonResponse({
                "sentiment": sentiment,
                "polarity": polarity,
            })

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON input."}, status=400)

    return JsonResponse({"error": "Invalid request method."}, status=405)

def dataset_analysis(request):
    analyzed_df = request.session.get("analyzed_df")

    if request.method == "POST" and request.FILES.get("dataset"):
        file = request.FILES["dataset"]
        filename = default_storage.save(file.name, file)
        filepath = os.path.join(default_storage.location, filename)

        try:
            if file.name.endswith(".csv"):
                df = pd.read_csv(filepath)
            elif file.name.endswith(".xlsx"):
                df = pd.read_excel(filepath)
            elif file.name.endswith(".txt"):
                df = pd.read_csv(filepath, sep='\t')
            else:
                raise ValueError("Unsupported file type")

            text_col = next((col for col in df.columns if any(keyword in col.lower() for keyword in ["text", "review", "content", "message", "comment", "description"])), None)
            if not text_col:
                raise ValueError("No suitable text column found")

            df["cleaned"] = df[text_col].astype(str).apply(clean_text)
            df["category"] = df["cleaned"].apply(classify_category)

            sentiments = []
            polarities = []
            for txt in df[text_col].astype(str):
                sent, pol, _ = analyze_sentiment(txt)
                sentiments.append(sent)
                polarities.append(pol)

            df["sentiment"] = sentiments
            df["polarity"] = polarities

            # Save result and charts
            df_result = df[[text_col, "sentiment", "polarity", "category"]]
            request.session["analyzed_df"] = df_result.to_dict(orient="records")
            request.session["text_col"] = text_col

            sentiment_counts = df["sentiment"].value_counts().to_dict()
            category_counts = df["category"].value_counts().to_dict()

            pie_fig, pie_ax = plt.subplots()
            pie_ax.pie(sentiment_counts.values(), labels=sentiment_counts.keys(), autopct='%1.1f%%', startangle=90)
            pie_ax.axis('equal')
            pie_chart = get_image_base64(pie_fig)
            plt.close(pie_fig)

            bar_fig, bar_ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x=list(category_counts.keys()), y=list(category_counts.values()), ax=bar_ax)
            bar_ax.set_xticklabels(bar_ax.get_xticklabels(), rotation=30)
            bar_chart = get_image_base64(bar_fig)
            plt.close(bar_fig)

            # Paginate
            paginator = Paginator(df_result.to_dict(orient="records"), 50)
            page_number = request.GET.get("page", 1)
            page_obj = paginator.get_page(page_number)

            return render(request, "sentiment_app/dataset_analysis.html", {
                "page_obj": page_obj,
                "pie_chart": pie_chart,
                "bar_chart": bar_chart,
                "filename": file.name,
                "show_table": True
            })

        except Exception as e:
            print("Error:", str(e))
            return render(request, "sentiment_app/dataset_analysis.html", {"error": str(e)})

    elif request.method == "GET" and analyzed_df:
        text_col = request.session.get("text_col", "Text")
        df_result = pd.DataFrame(analyzed_df)

        paginator = Paginator(df_result.to_dict(orient="records"), 50)
        page_number = request.GET.get("page", 1)
        page_obj = paginator.get_page(page_number)

        return render(request, "sentiment_app/dataset_analysis.html", {
            "page_obj": page_obj,
            "show_table": True
        })

    return render(request, "sentiment_app/dataset_analysis.html")


def download_csv(request):
    """Download the analyzed dataset as a CSV."""
    analyzed_df = request.session.get("analyzed_df")

    if analyzed_df is None:
        return HttpResponse("No dataset found to download.", status=400)

    # Convert from JSON to DataFrame
    df = pd.DataFrame(analyzed_df)

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="analyzed_dataset.csv"'
    df.to_csv(path_or_buf=response, index=False)

    return response