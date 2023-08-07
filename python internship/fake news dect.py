import tkinter as tk
from tkinter import filedialog
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error: {e}")
        return None

def preprocess_data(df):
    df = df.dropna(subset=['text', 'label'])
    return df

def train_model(df):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X = tfidf_vectorizer.fit_transform(df['text'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")

    return model, tfidf_vectorizer

def analyze_fake_news(file_path):
    df = load_data(file_path)
    if df is None:
        return None, None, "Error: Unable to load data from the CSV file."

    df = preprocess_data(df)
    if df.empty:
        return None, None, "Error: The CSV file does not contain 'text' and 'label' columns."

    model, tfidf_vectorizer = train_model(df)
    return model, tfidf_vectorizer, "File analyzed successfully!"

def is_fake_news(news_text, model, tfidf_vectorizer):
    X = tfidf_vectorizer.transform([news_text])
    prediction = model.predict(X)[0]
    return "Fake" if prediction == 1 else "Real"

def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[('CSV Files', '*.csv')])
    if file_path:
        model, tfidf_vectorizer, message = analyze_fake_news(file_path)
        analyze_label.config(text=message)
        if model and tfidf_vectorizer:
            analyze_btn.config(state=tk.NORMAL)
            global current_model, current_tfidf_vectorizer
            current_model, current_tfidf_vectorizer = model, tfidf_vectorizer
        else:
            analyze_btn.config(state=tk.DISABLED)

def analyze():
    news_text = news_text_entry.get("1.0", tk.END).strip()
    if not news_text:
        result_label.config(text="Please enter the news text to analyze.")
        return

    if not current_model or not current_tfidf_vectorizer:
        result_label.config(text="Please analyze a CSV file first.")
        return

    result = is_fake_news(news_text, current_model, current_tfidf_vectorizer)
    result_label.config(text=f"The news is {result}.")

# GUI setup
app = tk.Tk()
app.title("Fake News Detection")

browse_btn = tk.Button(app, text="Browse CSV File", command=browse_file)
browse_btn.pack(pady=10)

analyze_label = tk.Label(app, text="")
analyze_label.pack()

news_text_entry = tk.Text(app, width=50, height=10)
news_text_entry.pack(padx=10, pady=5)

analyze_btn = tk.Button(app, text="Analyze", command=analyze, state=tk.DISABLED)
analyze_btn.pack(pady=10)

result_label = tk.Label(app, text="")
result_label.pack()

current_model, current_tfidf_vectorizer = None, None

app.mainloop()
