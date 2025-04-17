import streamlit as st
import pandas as pd
from datetime import datetime
import os
import matplotlib.pyplot as plt

# Add other components of your app
st.title("AshaAI Chatbot 💙")
st.write("Welcome to AshaAI! Let's chat.")

menu = st.sidebar.radio("AshaAI Menu", [
    "New Chat",
    "Chat history",
    "Search chats",
    "give feedback",
    "admin dashboard",
    "about ashaai"
])

if menu == "give feedback":
    emoji_map = {
        1: "😞",
        2: "☹",
        3: "😐",
        4: "😊",
        5: "😁"
    }
    # Collect input
    rating = st.slider("Rate AshaAI (1-5)", 1, 5)
    # Display emoji based on slider value
    st.markdown(f"Your Rating: {emoji_map[rating]}")
    comment = st.text_input("Your thoughts(optional)")
    email = st.text_input("Email (optional)")
    feature = st.selectbox("Which part are you giving feedback for?",
                          ["Resume Parser", "Job Matching", "Motivation", "Course Suggestions", "Interactiveness", "Overall chat experience"])

    # Save to CSV
    if st.button("Submit Feedback"):
        new_feedback = pd.DataFrame({
            'timestamp': [datetime.now()],
            'rating': [rating],
            'comment': [comment],
            'user_email': [email],
            'feature': [feature],
        })
        if os.path.exists("feedback.csv"):
            new_feedback.to_csv("feedback.csv", mode='a', header=False, index=False)
        else:
            new_feedback.to_csv("feedback.csv", index=False)
        st.success("🎉 Thank you for your feedback!")

elif menu == "admin dashboard":
    admin_email = st.text_input("Enter Admin Email to access Dashboard")
    if "@ashaai.com" in admin_email:
        st.subheader("-----Welcome ADMIN! ✨-----")

        # Ensure feedback.csv exists before reading it
        if os.path.exists("feedback.csv"):
            df = pd.read_csv("feedback.csv")
            
            # Ensure 'rating' column is numeric and handle missing values
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
            
            st.header("📊 Feedback Summary")
            st.metric("Total Feedbacks", len(df))
            
            if not df.empty:
                avg_rating = round(df['rating'].mean(), 2)
            else:
                avg_rating = 0
            st.metric("Average Rating", avg_rating)

            # Histogram
            fig, ax = plt.subplots()
            ax.hist(df['rating'], bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], edgecolor='black', color='purple')
            ax.set_title("AshaAI User Ratings")
            ax.set_xlabel("Rating")
            ax.set_ylabel("Frequency")
            ax.set_xticks([1, 2, 3, 4, 5])
            st.pyplot(fig)

            # View full table
            st.subheader("All Feedback Entries")
            st.dataframe(df)

            # Delete feedback by row number
            row_to_delete = st.number_input("Enter row number to delete (0-based)", min_value=0, max_value=len(df)-1)
            if st.button("Delete Entry"):
                df = df.drop(index=row_to_delete)
                df.to_csv("feedback.csv", index=False)
                st.success("Deleted Successfully!")

            if st.button("Refresh"):
                st.experimental_rerun()
        else:
            st.warning("No feedback available. Please submit feedback first.")

    else:
        st.warning("Access Denied. ADMIN ONLY..")

elif menu == "New Chat":
    st.write("NEW CHAT!!!")

elif menu == "Chat history":
    st.write("Chat history pops out")

elif menu == "Search chats":
    st.write("Search history chats")

elif menu == "about ashaai":
    st.markdown("Display a few lines about the bot’s mission, built by Nidhi 💛")
