import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# ----------------------------
# Title
# ----------------------------
st.title("🚚 Packers and Movers Sentiment Analysis")

st.write("Customer Review Sentiment Analysis Dashboard")

# ----------------------------
# Load Dataset
# ----------------------------
data = pd.read_csv("reviews.csv")

# ----------------------------
# Create Sentiment
# ----------------------------
def sentiment(rating):
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"

data["Sentiment"] = data["Rating"].apply(sentiment)

# ----------------------------
# Show Dataset
# ----------------------------
st.subheader("Dataset")

st.dataframe(data)

# ----------------------------
# Sentiment Count
# ----------------------------
st.subheader("Sentiment Count")

st.write(data["Sentiment"].value_counts())

# ----------------------------
# Bar Chart
# ----------------------------
st.subheader("Bar Chart")

fig, ax = plt.subplots()

data["Sentiment"].value_counts().plot(kind="bar", ax=ax)

st.pyplot(fig)

# ----------------------------
# Pie Chart
# ----------------------------
st.subheader("Pie Chart")

fig2, ax2 = plt.subplots()

data["Sentiment"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    ax=ax2
)

ax2.set_ylabel("")

st.pyplot(fig2)

# ----------------------------
# Machine Learning
# ----------------------------
vectorizer = CountVectorizer()

X = vectorizer.fit_transform(data["Review"])

y = data["Sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = MultinomialNB()

model.fit(X_train, y_train)

# ----------------------------
# Prediction
# ----------------------------
st.subheader("Predict Customer Review")

review = st.text_input("Enter Customer Review")

if st.button("Predict"):

    review_vector = vectorizer.transform([review])

    result = model.predict(review_vector)

    st.success(result[0])
    # ==========================================
# Sidebar - Project Information
# ==========================================

st.sidebar.title("📌 Project Information")

st.sidebar.write("""
**Project Name:**
Customer Review Sentiment Analysis

**Domain:**
Data Analytics & Machine Learning

**Objective:**
Analyze customer reviews and predict whether
they are Positive, Neutral, or Negative.

**Technologies Used:**
- Python
- Pandas
- Matplotlib
- Scikit-learn
- Streamlit

**Machine Learning Algorithm:**
Multinomial Naive Bayes

**Dataset:**
Customer Reviews Dataset
""")

# ==========================================
# Sidebar - Developer Information
# ==========================================

st.sidebar.title("👩‍💻 Developer Information")

st.sidebar.write("""
**Developer Name:**
Manisha Suresh

**Degree:**
B.E. Computer Science and Engineering

**Internship Project**

**Project Version:**
1.0

**Developed Using:**
Python & Streamlit
""")

# ==========================================
# Sidebar - Contact Information (Optional)
# ==========================================

st.sidebar.title("📞 Contact")

st.sidebar.write("""
📧 Email:manishasure2005@gmail.com

🌐 LinkedIn:https://www.linkedin.com/in/manisha-suresh-8063a9307?utm_source=share_via&utm_content=profile&utm_medium=member_android


📍 Location: kattupakam,chennai,Tamil Nadu, India
""")