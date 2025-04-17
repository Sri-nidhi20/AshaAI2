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
    #collect input
    rating = st.slider("Rate AshaAI (1-5)",1, 5)
    #display emoji based on slider value
    st.markdown(f"Your Rating: {emoji_map[rating]}")
    comment = st.text_input("Your thoughts(optional)")
    email = st.text_input("Email (optional)")
    feature = st.selectbox("Which part are you giving feedback for?",
                          ["Resume Parser", "Job Matching", "Motivation", "Course Suggestions", "Interactiveness", "Overall chat experience"])
    #save to CSV
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
#admin dashboard
elif menu == "admin dashboard":
    admin_email = st.text_input("Enter Admin Email to access Dashboard")
    if "@ashaai.com" in admin_email:
        st.subheader("-----Welcome ADMIN! ✨-----")
        df = pd.read_csv("feedback.csv")
        st.header("📊 Feedback Summary")
        st.metric("Total Feedbacks", len(df))
        st.metric("Average Rating", round(df['rating'].mean(), 2))
        #histogram
        fig, ax = plt.subplots()
        ax.hist(df['rating'], bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], edgecolor='black', color='purple')
        ax.set_title("AshaAI User Ratings")
        ax.set_xlabel("Rating")
        ax.set_ylabel("Frequency")
        ax.set_xticks([1, 2, 3, 4, 5])
        st.pyplot(fig)
        #view full table
        st.subheader("All Feedback Entries")
        st.dataframe(df)
        #delete feedback by row number
        row_to_delete = st.number_input("Enter row number to delete (0-based)", min_value=0, max_value=len(df)-1)
        if st.button("Delete Entry"):
            df = df.drop(index=row_to_delete)
            df.to_csv("feedback.csv", index=False)
            st.success("Deleted Successfully!")
            if st.button("Refresh"):
                st.experimental_rerun()
    else:
        st.warning("Access Denied. ADMIN ONLY..")
#new chat
elif menu == "New Chat":
    st.write("NEW CHAT!!!")
#chat history
elif menu == "Chat history":
    st.write("Chat history pops out")
#search chats
elif menu == "Search chats":
    st.write("Search history chats")
#about asha ai
elif menu == "about ashaai":
    st.markdown("Display a few lines about the bot’s mission, built by Nidhi 💛")
