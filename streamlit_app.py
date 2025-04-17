import streamlit as st
import pandas as pd
from datetime import datetime
import os
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
import requests
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# File path for feedback (works even in restricted environments like Streamlit Cloud)
feedback_file = "/tmp/feedback.csv"

# Title and welcome
st.title("AshaAI Chatbot 💙")
st.write("Welcome to AshaAI! Let's chat.")

# Sidebar menu
menu = st.sidebar.radio("AshaAI Menu", [
    "New Chat ➕",
    "Chat History 🗨",
    "Search Chats 🔍",
    "Give Feedback 😊😐☹️",
    "Admin Dashboard 📊",
    "About AshaAI 👩🤖"
])

# Give feedback
if menu == "Give Feedback 😊😐☹️":
    emoji_map = {
        1: "😞",
        2: "😕",
        3: "😐",
        4: "😊",
        5: "😁"
    }

    rating = st.slider("Rate AshaAI (1-5)", 1, 5)
    st.markdown(f"Your Rating: {emoji_map[rating]}")
    comment = st.text_input("Your thoughts (optional)")
    email = st.text_input("Email (optional)")
    feature = st.selectbox("Which part are you giving feedback for?", [
        "Resume Parser", "Job Matching", "Motivation", "Course Suggestions", "Interactiveness", "Overall chat experience"
    ])

    if st.button("Submit Feedback"):
        new_feedback = pd.DataFrame({
            'timestamp': [datetime.now().strftime("%d-%b-%Y")],
            'rating': [rating],
            'comment': [comment],
            'user_email': [email],
            'feature': [feature],
        })
        if os.path.exists(feedback_file):
            new_feedback.to_csv(feedback_file, mode='a', header=False, index=False)
        else:
            new_feedback.to_csv(feedback_file, index=False)
        lottie_success = load_lottieurl("https://lottie.host/embed/73099729-1a0e-4d04-b13c-129c1e9ad3aa/WEt2VVYpfi.json")
        st.success("🎉 Thank you for your feedback!🤗🤩")
        st_lottie(lottie_success, height=200, key="success")
        
# Admin Dashboard
elif menu == "Admin Dashboard 📊":
    admin_email = st.text_input("Enter Admin Email to access Dashboard")
    if "@ashaai.com" in admin_email:
        st.subheader("-----Welcome ADMIN! ✨-----")
        if os.path.exists(feedback_file):
            df = pd.read_csv(feedback_file)
            st.header("📊 Feedback Summary")
            st.metric("Total Feedbacks", len(df))
            st.metric("Average Rating", round(df['rating'].mean(), 2))

            fig, ax = plt.subplots()
            ax.hist(df['rating'], bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], edgecolor='black', color='purple')
            ax.set_title("AshaAI User Ratings")
            ax.set_xlabel("Rating")
            ax.set_ylabel("Frequency")
            ax.set_xticks([1, 2, 3, 4, 5])
            st.pyplot(fig)

            st.subheader("All Feedback Entries")
            st.dataframe(df)

            row_to_delete = st.number_input("Enter row number to delete (0-based)", min_value=0, max_value=len(df)-1)
            if st.button("Delete Entry"):
                df = df.drop(index=row_to_delete)
                df.to_csv(feedback_file, index=False)
                st.success("Deleted Successfully!")
                if st.button("Refresh"):
                    st.experimental_rerun()
        else:
            st.warning("⚠️ No feedback data available yet!")
    else:
        st.warning("Access Denied. ADMIN ONLY..")

# New Chat
elif menu == "New Chat ➕":
    st.write("NEW CHAT!!!")

# Chat History
elif menu == "Chat History 🗨":
    st.write("Chat history pops out")

# Search Chats
elif menu == "Search Chats 🔍":
    st.write("Search history chats")

# About
elif menu == "About AshaAI 👩🤖":
    st.markdown("Display a few lines about the bot’s mission, built by Nidhi 💛")
