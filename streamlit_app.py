# streamlit_app.py

import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os
import requests
import time
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
from PIL import Image

# ------------------ CONFIG ------------------ #
st.set_page_config(page_title="AshaAI Chatbot", layout="wide")
genai.configure(api_key = st.secrets["GOOGLE_API_KEY"])
model = genai.GenerativeModel("gemini-pro")
feedback_file = "feedback.csv"

# ------------------ UTILS ------------------ #
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def query_gemini(prompt):
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# ------------------ HEADER ------------------ #
logo = Image.open("ashaai_logo.jpg")
st.image(logo, width=150)
st.markdown("<h2 style='text-align:center;'>Welcome to AshaAI 💙 - your Career Companion</h2>", unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------ #
menu = st.sidebar.radio("AshaAI Menu", [
    "New Chat ➕",
    "Chat History 🗨",
    "Search Chats 🔍",
    "Give Feedback 😊😐🙁",
    "Admin Dashboard 📊",
    "About AshaAI 👩‍🤖"
])

# ------------------ NEW CHAT ------------------ #
if menu == "New Chat ➕":
    st.subheader("💬 Start Chatting with AshaAI")

    # Initialize session state variables
    if "chat" not in st.session_state:
        st.session_state.chat = []
    if "pending_input" not in st.session_state:
        st.session_state.pending_input = None

    # Display chat messages
    for sender, msg in st.session_state.chat:
        if sender == "user":
            st.markdown(f"**👩 You:** {msg}")
        else:
            st.markdown(f"**🤖 AshaAI:** {msg}")

    # Capture new input
    user_input = st.chat_input("Your Question...")
    if user_input:
        st.session_state.pending_input = user_input

    # Respond to pending input
    if st.session_state.pending_input:
        st.session_state.chat.append(("user", st.session_state.pending_input))
        with st.spinner("AshaAI is thinking..."):
            reply = query_gemini(st.session_state.pending_input)
        st.session_state.chat.append(("AshaAI", reply))
        st.session_state.pending_input = None

# ------------------ CHAT HISTORY ------------------ #
elif menu == "Chat History 🗨":
    st.subheader("📜 Chat History")
    if "chat" in st.session_state and st.session_state.chat:
        for role, text in st.session_state.chat:
            if role == "user":
                st.markdown(f"**👩 You:** {text}")
            else:
                st.markdown(f"**🤖 AshaAI:** {text}")
    else:
        st.info("No previous chats available yet.")

# ------------------ FEEDBACK ------------------ #
elif menu == "Give Feedback 😊😐🙁":
    st.subheader("📝 Share your feedback!")
    emoji_map = {1: "😞", 2: "😕", 3: "😐", 4: "😊", 5: "😁"}
    rating = st.slider("Rate AshaAI (1-5)", 1, 5)
    st.markdown(f"Your Rating: {emoji_map[rating]}")
    comment = st.text_input("Your thoughts (optional)")
    email = st.text_input("Email (optional)")
    feature = st.selectbox("Which part are you giving feedback for?", [
        "Resume Parser", "Job Matching", "Motivation", "Course Suggestions", "Interactiveness", "Overall chat experience"])

    if st.button("Submit Feedback"):
        new_feedback = pd.DataFrame({
            'timestamp': [datetime.now().strftime("%d-%b-%Y %H:%M:%S")],
            'rating': [rating],
            'comment': [comment],
            'user_email': [email],
            'feature': [feature],
        })
        if os.path.exists(feedback_file):
            new_feedback.to_csv(feedback_file, mode='a', header=False, index=False)
        else:
            new_feedback.to_csv(feedback_file, index=False)
        lottie_success = load_lottieurl("https://lottie.host/a18d8f78-7a96-4938-a960-11846b793789/iUFyJaP2CZ.json")
        st.success("🎉 Thank you for your feedback! 🤗🤩")
        st_lottie(lottie_success, height=500, key="success")

# ------------------ ADMIN DASHBOARD ------------------ #
elif menu == "Admin Dashboard 📊":
    st.subheader("🛠️ Admin Dashboard")
    admin_email = st.text_input("Enter Admin Email to access Dashboard")
    if "@ashaai.com" in admin_email:
        if os.path.exists(feedback_file):
            df = pd.read_csv(feedback_file)
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
            row_to_delete = st.number_input("Enter row number to delete (0-based)", min_value = 0,max_value = len(df)-1)
            if st.button("Delete Entry"):
                df = df.drop(index = row_to_delete)
                df.to_csv("feedback.csv", index = False)
                st.success("Deleted Successfully!!")
                if st.button("Refresh.."):
                    st.experimental.rerun()
        else:
            st.warning("⚠️ No feedback data available yet.")
    else:
        st.warning("Access Denied. ADMIN ONLY..")

# ------------------ SEARCH CHATS (Coming Soon) ------------------ #
elif menu == "Search Chats 🔍":
    st.subheader("🔍 Search Your Chats (Coming Soon)")

# ------------------ ABOUT ------------------ #
elif menu == "About AshaAI 👩‍🤖":
    st.subheader("About AshaAI")
    st.markdown("""
    AshaAI is a personalized career guidance chatbot designed to support women in their professional journeys.
    Whether it's job matching, resume guidance, emotional motivation, or mentorship — AshaAI is your friendly, always-there assistant. 🤖💛

    Built with love and purpose by **Nidhi** for the ASHA AI Hackathon 2025. ✨
    """)

