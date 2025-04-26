import streamlit as st
import google.generativeai as genai
import os
import requests
import random
import pandas as pd
from datetime import datetime, date
timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
from PIL import Image
import time
import json
import logging
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from urllib.parse import quote_plus
from PyPDF2 import PdfReader
from wordcloud import WordCloud
import json
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except Exception:
    try:
        nltk.download('vader_lexicon', quiet = True)
    except Exception as e:
        logging.error(f"[{timestamp}] Error downloading VADER lexicon: {e}")

# ------------------ CONFIG ------------------ #
st.set_page_config(page_title="AshaAI Chatbot", layout="wide")
genai.configure(api_key=st.secrets.get("GEMINI_API_KEY"))
model = genai.GenerativeModel("models/gemini-2.0-flash")
feedback_file = "feedback.csv"
history_file = "chat_history.json"
logging.basicConfig(filename = "ashaai_log.txt", level=logging.INFO)

#--------------------------defining quiz data -----------------------------#
def get_gemini_answer(question_text):
    prompt_text = f"Answer the following question related to programming concisely, within 2000 characters: {question_text}"
    try:
        # Sending query to Gemini API
        response = model.generate_content([{"parts": [{"text": prompt_text}]}])
        if response.text:
            # Trimming response if necessary
            trimmed_response = response.text[:2000]
            return trimmed_response
        else:
            return "I'm sorry, I couldn't provide an answer at the moment."
    except Exception as e:
        logging.error(f"Error fetching answer from Gemini: {e}")
        return "I'm having trouble retrieving the answer right now."

def load_quiz_data():
    try:
        with open("quiz_data.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"[{timestamp}] quiz_data.json file not found.")
        return {}  # Return an empty dictionary if the file is not found
    except json.JSONDecodeError as e:
        logging.error(f"[{timestamp}] Error decoding quiz_data.json: {e}")
        return {}

quiz_data = load_quiz_data()
if not quiz_data:
    st.warning("⚠️ Quiz data could not be loaded. Please ensure 'quiz_data.json' exists and is valid.")
elif "language" in st.session_state and st.session_state.language and st.session_state.language not in quiz_data:
    st.warning(f"⚠️ No quiz questions found for the language: {st.session_state.language}")
    st.session_state.language = None # Reset language to avoid further errors

#========== Session State======
if "streak" not in st.session_state:
    st.session_state.streak = 0
if "last_played" not in st.session_state:
    st.session_state.last_played = ""
if "language" not in st.session_state:
    st.session_state.language = None
if "difficulty" not in st.session_state:
    st.session_state.difficulty = None
if "questions" not in st.session_state:
    st.session_state.questions = []
if "score" not in st.session_state:
    st.session_state.score = 0
if "answered_today" not in st.session_state:
    st.session_state.answered_today = False
if "user_answers" not in st.session_state:
    st.session_state.user_answers = []
if "quiz_started" not in st.session_state:
    st.session_state.quiz_started = False

# ------------------ UTILS ------------------ #
SAVE_DIR = "saved_chats"

def save_chat(chat_history, chat_name):
    os.makedirs(SAVE_DIR, exist_ok=True)
    filename = f"{chat_name.replace(' ', '_')}.json"
    filepath = os.path.join(SAVE_DIR, filename)
    try:
        with open(filepath, 'w') as f:
            json.dump(chat_history, f)
        st.success(f"Chat saved as '{chat_name}'!")
    except Exception as e:
        st.error(f"Error saving chat: {e}")

def load_saved_chat(chat_name):
    filename = f"{chat_name.replace(' ', '_')}.json"
    filepath = os.path.join(SAVE_DIR, filename)
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading chat '{chat_name}': {e}")
        return None

def get_saved_chat_names():
    os.makedirs(SAVE_DIR, exist_ok=True)
    names = []
    for filename in os.listdir(SAVE_DIR):
        if filename.endswith(".json"):
            names.append(filename[:-5].replace('_', ' '))
    return names
    
if "selected_chat" not in st.session_state:
        st.session_state.selected_chat = None
    
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def query_gemini(prompt_text, timeout_seconds=60):
    logging.info(f"[{timestamp}] User prompt: {prompt_text}")
    refined_prompt_text = f"Answer the following questions concisely and completely within 2000 characters: {prompt_text}. Please prioritize finishing your thought or explanation within the character limit, even if it means covering slightly less ground."

    name = None
    if re.match(r"^myself\s+([A-Za-z]+)(?:\s|$)", prompt_text, re.IGNORECASE):
        name = re.match(r"^myself\s+([A-Za-z]+)(?:\s|$)", prompt_text, re.IGNORECASE).group(1)
    elif re.match(r"^I am\s+([A-Za-z]+)(?:\s|$)", prompt_text, re.IGNORECASE):
        name = re.match(r"^I am\s+([A-Za-z]+)(?:\s|$)", prompt_text, re.IGNORECASE).group(1)
    elif re.match(r"^my name is\s+([A-Za-z]+)(?:\s|$)", prompt_text, re.IGNORECASE):
        name = re.match(r"^my name is\s+([A-Za-z]+)(?:\s|$)", prompt_text, re.IGNORECASE).group(1)

    if name:
        st.session_state.user_profile = st.session_state.get("user_profile", {})
        st.session_state.user_profile["name"] = name
        return f"Hello {name}! It's nice to meet you. How can I help you with your career journey today?"

    greetings = r"^(hello|hi|hey|greetings|good morning|good afternoon|good evening)\b.*"
    if re.match(greetings, prompt_text, re.IGNORECASE):
        if "user_profile" in st.session_state and "name" in st.session_state.user_profile:
            return f"Hello {st.session_state.user_profile['name']}! How can I assist you with your career journey today?"
        else:
            st.session_state.is_career_context = False
            return "Hello there! How can I assist you with your career journey today?"

    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(prompt_text)
    logging.info(f"[{timestamp}] Sentiment scores: {vs}")

    if vs['compound'] < -0.2:
        encouragement_query = f"The user is expressing negative feelings like '{prompt_text}'. Offer a short, encouraging and supportive message related to career challenges."
        logging.info(f"[{timestamp}] Negative sentiment detected. Sending encouragement query.")
        try:
            contents = [{"parts": [{"text": encouragement_query}]}]
            response = model.generate_content(contents)
            if response.text:
                st.session_state.is_career_context = True
                return response.text[:2000]
            else:
                st.session_state.is_career_context = True
                return "Sending you some positive vibes! Remember, career journeys have ups and downs. How can I help you?"
        except Exception as e:
            logging.error(f"[{timestamp}] Error fetching encouragement: {e}")
            return f"Error: {str(e)}"

    motivation_keywords = r"\b(motivate|inspire|inspiration|encouragement|uplift|positive outlook|give me motivation)\b"
    if re.search(motivation_keywords, prompt_text, re.IGNORECASE):
        motivation_query = f"Give a short, inspiring message related to {prompt_text.lower().replace('give me motivation', '').strip()}."
        logging.info(f"[{timestamp}] Sending motivation query.")
        try:
            contents = [{"parts": [{"text": motivation_query}]}]
            response = model.generate_content(contents)
            if response.text:
                st.session_state.is_career_context = True
                return response.text[:2000]
            else:
                st.session_state.is_career_context = True
                return "Here's a little something to keep you going: Every challenge is an opportunity to grow!"
        except Exception as e:
            logging.error(f"[{timestamp}] Error fetching motivation: {e}")
            return "I'm experiencing a slight delay. Please try again."

    career_keywords_initial = r"\b(...your big list here...)\b"  # keep your full big list
    career_skills_query = r"\b(...your skill query patterns...)\b"  # keep your full patterns

    if "is_career_context" not in st.session_state:
        st.session_state.is_career_context = False
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = {}

    if re.search(r"\b(skills|requirements|career path|job market|industry trends)\b", prompt_text, re.IGNORECASE):
        st.session_state.user_profile["intent"] = "career_exploration"
    elif re.search(r"\b(jobs|opportunities|hiring|placement|recruitment|internship)\b", prompt_text, re.IGNORECASE):
        st.session_state.user_profile["intent"] = "job_search"
    elif re.search(r"\b(resume|cv|cover letter|portfolio|linkedin)\b", prompt_text, re.IGNORECASE):
        st.session_state.user_profile["intent"] = "application_help"
    elif re.search(r"\b(learn|training|upskill|reskill|courses|certifications)\b", prompt_text, re.IGNORECASE):
        st.session_state.user_profile["intent"] = "learning_development"

    keywords = re.findall(r"\b(\w+)\b", prompt_text.lower())
    st.session_state.user_profile["topics"] = st.session_state.user_profile.get("topics", set()).union(keywords)

    logging.info(f"[{timestamp}] User Profile: {st.session_state.user_profile}")

    if re.search(career_keywords_initial, prompt_text, re.IGNORECASE) or re.search(career_skills_query, prompt_text, re.IGNORECASE):
        logging.info(f"[{timestamp}] Career-related query detected.")
        st.session_state.is_career_context = True
        try:
            contents = [{"parts": [{"text": refined_prompt_text}]}]
            response = model.generate_content(contents)
            if response.text:
                relevant_text = response.text[:2000]
                lines = relevant_text.split(". ")
                formatted_text = "\n".join(f"* {line.strip()}." for line in lines if line.strip())
                return formatted_text.strip()
            else:
                return "Hmm, I didn't get a clear response for that career query. Could you please rephrase?"
        except Exception as e:
            logging.error(f"[{timestamp}] Error handling career query: {e}")
            return "Sorry! I'm facing difficulty processing your request."

    # 🔥🔥 FINAL FALLBACK 🔥🔥
    logging.info(f"[{timestamp}] No match found. Returning fallback response.")
    return "I'm specialized in assisting with career guidance. Could you please ask something related to your career journey?"

#------------------ defining Job search api -----------------#
APP_ID = st.secrets["adzuna"]["APP_ID"]
APP_KEY = st.secrets["adzuna"]["APP_KEY"]
BASE_URL = "https://api.adzuna.com/v1/api/jobs/gb/search/1"

#------------------ Helper functions---------------------#
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    resume_text = ""
    for page in pdf_reader.pages:
        text = page.extract_text()
        if text:
            resume_text += text
    return resume_text

def extract_skills(text):
    skill_keywords = [
        "python", "java", "c++", "data analysis", "machine learning", "deep learning", "AWS", "Azure", "GCP", "CI/CD", "Kubernetes", "Terraform", "prompt engineering", 
        "sql", "excel", "communication", "leadership", "problem solving", "javascript" "react", "angular", "Vue.js", "full-stack development", "Tableau", 
        "teamwork", "project management", "tensorflow", "pandas", "numpy", "django", "Power BI", "Tensorflow", "pytorch", "cloud security", "cybersecurity fundamentals" 
    ]
    skills_found = [skill.title() for skill in skill_keywords if skill.lower() in text.lower()]
    return skills_found

def detect_experience(text):
    exp_patterns = re.findall(r'(\d+)\+?\s*(years|year)', text.lower())
    if exp_patterns:
        years = [int(y[0]) for y in exp_patterns]
        return max(years)
    return 0

def detect_degree(text):
    degrees = ["Bachelor", "Master", "PhD", "B.tech", "M.Tech", "MBA"]
    for degree in degrees:
        if degree.lower() in text.lower():
            return degree
    return "Not Detected"

def generate_score(skills, experience, degree):
    score = 0
    if skills:
        score += 30
    if experience >= 2:
        score += 30
    if degree != "Not Detected":
        score += 30
    score += 10 
    return score

def suggest_next_steps(skills, experience):
    suggestions = []
    if len(skills) < 3:
        suggestion.append("Add more technical and soft skills.")
    if experience < 2:
        suggestions.append("Gain internship or project experience.")
    if not suggestions:
        suggestions.append("Your resume looks strong! Keep applying confidently!")
    return suggestions
def award_badges(skills):
    badges = []
    if "Python" in skills:
        badges.append("🏅 Python Pro")
    if any(skill in skills for skill in ["Machine Learning", "Data Analysis"]):
        badges.append("🏅 Data Enthusiast")
    if "Leadership" in skills:
        badges.append("🏅 Team Leader")
    if "Communication" in skills:
        badges.append("🏅 Excellent Communicator")
    if "Full-Stack Development" in skills or any(skill in skills for skill in ["React", "Angular", "Vue.js"]):
        badges.append("🏅 Frontend Champ")
    if any(skill in skills for skill in ["AWS", "Azure", "GCP"]):
        badges.append("🏅 Cloud Explorer")
    if "Cybersecurity Fundamentals" in skills:
        badges.append("🏅 Security Aware")
    return badges
def generate_wordcloud(skills):
    text = " ".join(skills)
    wordcloud = WordCloud(width = 800, height = 400, background_color = 'pink').generate(text)
    return wordcloud
# ------------------ HEADER ------------------ #
try:
    logo = Image.open("ashaai_logo.jpg")
    st.image(logo, width=150)
except FileNotFoundError:
    st.warning("⚠ 'ashaai_logo.jpg' not found in the current directory.")

st.markdown("<h2 style='text-align:center;'>Welcome to AshaAI 💙 - your Career Companion</h2>", unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------ #
menu = st.sidebar.radio("AshaAI Menu", [
    "New Chat ➕",
    "Chat History 🗨",
    "Search Chats 🔍",
    "Job Search 💼",
    "Resume Analysis 📄",
    "Give Feedback 😊😐🙁",
    "Admin Dashboard 📊",
    "About AshaAI 👩‍🤖",
    "QUIZ TIME 🤩🥳"
])

# ------------------ NEW CHAT ------------------ #
if menu == "New Chat ➕":
    st.subheader("💬 Start Chatting with AshaAI")

    if "chat" not in st.session_state:
        st.session_state.chat = []
    if "pending_input" not in st.session_state:
        st.session_state.pending_input = None
    if "chat_turn" not in st.session_state:
        st.session_state.chat_turn = 0
    if "is_career_context" not in st.session_state:
        st.session_state.is_career_context = False
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = {}
    if "chat_history_name" not in st.session_state:
        st.session_state.chat_history_name = f"Chat {st.session_state.chat_turn}"


    for i, (sender, msg) in enumerate(st.session_state.chat):
        if sender == "user":
            st.markdown(f"👩‍💼 You:** {msg}")
        else:
            st.markdown(f"👩 AshaAI:** {msg}")
    col1, col2 = st.columns([3, 1]) 

    with col1:
        edited_chat_name = st.text_input("Chat Name:", st.session_state.chat_history_name)
        st.session_state.chat_history_name = edited_chat_name

    with col2:
        if st.button("➕ New Chat", use_container_width=True):
            st.session_state.chat = []
            st.session_state.pending_input = None
            st.session_state.chat_turn = 0
            st.session_state.is_career_context = False
            st.session_state.user_profile = {}
            st.session_state.chat_history_name = f"Chat {st.session_state.chat_turn + 1}" 
            st.rerun() 

    if st.button("Save Chat"):
        if st.session_state.chat:
            save_chat(st.session_state.chat, st.session_state.chat_history_name)
        else:
            st.warning("No chat history to save.")
            
    user_input = st.chat_input("Your Question...")
    if user_input:
        st.markdown(f"👩‍💼 You:** {user_input}")
        st.session_state.chat.append(("user", user_input))
        st.session_state.pending_input = user_input
        st.session_state.chat_turn += 1

    if st.session_state.pending_input:
        prompt_text = st.session_state.pending_input
        logging.info(f"[{timestamp}] Prompt sent to query_gemini: {prompt_text}") # Log the prompt
        with st.spinner("AshaAI is thinking..."):
            reply = query_gemini(prompt_text)
        placeholder = st.empty()
        typed_response = ""
        error_phrases = [
            "couldn't provide an answer",
            "having trouble retrieving",
            "I'm experiencing a slight delay",
            "trouble processing your request",
            "please try again",
            "Sorry! I'm having trouble"
        ]

        if (reply is None) or any (err in reply for err in error_phrases):
            typed_response = (
                "❌ AshaAI is having trouble processing your request. Please try again in a few moments.\n\n"
                "Till then you can refer to the following stories 😉 \n\n"
                "**STORY 1: **\nOnce upon a time in a bustling city, Priya—a young woman with zero baking skills—decided to open a bakery. Her first attempt? Burnt muffins that could double as paperweights."
                "Instead of giving up, she advertised them as “Unbreakable Bond Muffins”—a hit among clumsy folks who needed a snack that wouldn’t crumble. Her honesty, humor, and determination made her bakery a sensation."
                "Priya proved that failure can be the best recipe for success. \n\n"
                "**STORY 2:**\n In a small town, Radha started a taxi service—unusual for a woman in her area. On her first day, she got a skeptical passenger who asked, “Can you even change a tire?” Radha replied with a grin,"
                "“I can change tires, change routes, and even change your mind about women drivers!” By the end of the ride, her charm and smooth driving turned that skeptic into her biggest advocate."
                "Radha's service thrived, proving that confidence and skill can break stereotypes with style!"
            )
            placeholder.markdown(f"👩 AshaAI:** {typed_response}")
            st.session_state.chat.append(("AshaAI", typed_response))
        else:
            for char in reply:
                typed_response += char
                placeholder.markdown(f"👩 AshaAI:** {typed_response}")
                time.sleep(0.01)
            placeholder.markdown(f"👩 AshaAI:** {typed_response}")
            st.session_state.chat.append(("AshaAI", reply))

        st.session_state.pending_input = None

# ------------------ CHAT HISTORY ------------------ #
elif menu == "Chat History 🗨":
    st.subheader("📜 Chat History")
    saved_chat_names = get_saved_chat_names()
    if saved_chat_names:
        st.session_state.selected_chat = st.selectbox("Select a chat:", saved_chat_names, placeholder = "Select One")
        if st.session_state.selected_chat:
            st.subheader(f"Chat: {st.session_state.selected_chat}")
            selected_chat_history = load_saved_chat(st.session_state.selected_chat)
            if selected_chat_history:
                for sender, msg in selected_chat_history:
                    if sender == "user":
                        st.markdown(f"👩‍💼 You:** {msg}")
                    else:
                        st.markdown(f"👩 AshaAI:** {msg}")
            else:
                st.info("Could not load selected chat.")
    else:
        st.info("No chats have been saved yet.")

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
        st_lottie(lottie_success, height=300, key="success_feedback") # Added a unique key

# ------------------ ADMIN DASHBOARD ------------------ #
elif menu == "Admin Dashboard 📊":
    st.subheader("🛠 Admin Dashboard")
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
            row_to_delete = st.number_input("Enter row number to delete (0-based)", min_value=0, max_value=len(df) - 1)
            if st.button("Delete Entry"):
                try:
                    df = df.drop(index=row_to_delete)
                    df.to_csv("feedback.csv", index=False)
                    st.success(f"Deleted row {row_to_delete} successfully!")
                except IndexError:
                    st.error("Invalid row number.")
        else:
            st.warning("⚠ No feedback data available yet.")
    else:
        st.warning("Access Denied. ADMIN ONLY..")

# ------------------ SEARCH CHATS (Coming Soon) ------------------ #
elif menu == "Search Chats 🔍":
    st.subheader("🔍 Search Your Chats")
    search_term = st.text_input("Enter chat name to search:")
    if search_term:
        found_chat = load_saved_chat(search_term)
        if found_chat:
            st.subheader(f"Conversation: {search_term}")
            for sender, msg in found_chat:
                if sender == "user":
                    st.markdown(f"👩‍💼 You:** {msg}")
                else:
                    st.markdown(f"👩 AshaAI:** {msg}")
        else:
            st.info(f"No chat found with the name: '{search_term}'")

# ------------------ ABOUT ------------------ #
elif menu == "About AshaAI 👩‍🤖":
    st.subheader("About AshaAI")
    st.markdown("""
    *AshaAI* is your personal career companion — an AI-powered chatbot designed *exclusively for women* to support, guide, and empower them on their professional journey. 💙

It helps you with:
- 🔍 Discovering job opportunities tailored to your skills and interests
- 📄 Resume insights and application tips
- 🎯 Personalized course & upskilling suggestions
- 💡 Motivation and career growth advice
- 👩‍🏫 Access to mentorship and community events
- 🌈 Gender-bias free, inclusive conversations

AshaAI remembers your previous chats and keeps conversations human-like — making career guidance feel as natural as talking to a friend.

Built by *Nidhi 💛* with love and purpose for the *ASHA AI Hackathon 2025*, AshaAI combines real tech with real empathy.

> “Asha” means hope — and that’s exactly what this AI brings to every woman’s career journey.

""")
# --------------------- QUIZ -------------------------#
elif menu == "QUIZ TIME 🤩🥳":
    st.header("It's the Quiz Time!!")
    st.subheader("🎯 Ready, Set, Code! 💻 Time to show off your skills and conquer this quiz like a coding pro! 💥")
    today = date.today()

    # Check if the quiz has already been answered today
    if st.session_state.last_played != today:
        st.session_state.answered_today = False

    # Display the current streak
    st.markdown(f"🔥 **Your Current Streak:** '{st.session_state.streak}' days")

    # Inform the user if they've already taken the quiz today
    if st.session_state.answered_today:
        st.success("✅ You've already taken today's quiz. Come Back Tomorrow to keep your streak alive! Till then keep practicing😉")
    else:
        if not st.session_state.get('quiz_started'):
            st.selectbox("Choose a programming language:", list(quiz_data.keys()), key="language")

            # Difficulty Selection (only if language is selected)
            if st.session_state.language:
                st.selectbox("Select difficulty level:", ["Easy", "Medium", "Hard"], key="difficulty")

            # Start Quiz Button (appears after language and difficulty are selected)
            if st.session_state.language and st.session_state.difficulty:
                if st.button("Start Quiz", key="start_quiz_button"):  # Added unique key
                    # Load questions if not already loaded
                    if not st.session_state.questions:
                        all_questions = quiz_data.get(st.session_state.language, {}).get(st.session_state.difficulty.lower(), [])
                        if len(all_questions) < 3:
                            st.error(f"⚠️ Not enough questions available for {st.session_state.language} at {st.session_state.difficulty} difficulty.")
                            st.stop()
                        st.session_state.questions = random.sample(all_questions, 3)
                        st.session_state.user_answers = [""] * 3  # Initialize as a list of empty strings
                        st.session_state.score = 0
                        st.session_state.quiz_started = True  # Flag to indicate quiz has started
                        st.session_state.answered_today = False  # Reset for fresh quiz day
                        st.rerun() # Still need rerun to show the questions
        else:
            for i, q in enumerate(st.session_state.questions):
                st.markdown(f"**Question {i+1}:** {q['question']}")
                st.session_state.user_answers[i] = st.text_area(
                    f"Your answer {i+1}", value=st.session_state.user_answers[i], key=f"user_answer_{i}"
                )

            # Submit Answers Button (appears after all questions are displayed)
            if st.button("Submit all answers"):
                with st.spinner("🧠 Evaluating your answers..."):
                    prompts = []
                    for i in range(3):
                        q = st.session_state.questions[i]
                        user_ans = st.session_state.user_answers[i]
                        prompt = f"""
                        Evaluate the user's programming answer.
                        Question: {q['question']}
                        User Answer: {user_ans}
                        Respond in JSON with:
                        - "is_correct": true/false
                        - "explanation": brief feedback or correction
                        """
                        prompts.append(prompt)

                    results = []
                    for p in prompts:
                        response = model.generate_content(p)  # This call should use your AI model
                        try:
                            json_text = response.text.strip().split("''")[-1]
                            result = json.loads(json_text)
                        except Exception as e:
                            result = {"is_correct": False, "explanation": "Couldn't evaluate the answer properly."}
                        results.append(result)
                    correct_count = 0
                    for i, r in enumerate(results):
                        st.markdown(f"*Q{i+1}:* {st.session_state.questions[i]['question']}")
                        if r["is_correct"]:
                            st.success(f"✅ Correct!!")
                            correct_count += 1
                        else:
                            st.error(f"❌ Incorrect.. {r['explanation']}")

                    # Display Results and Update Streak
                    if correct_count == 3:
                        st.balloons()
                        st.success("🥳💃 Perfect Score!! You're on fire Buddy! Keep it up🤗")
                        st.session_state.streak += 1
                    else:
                        st.warning(f"You got {correct_count}/3 correct. Keep practicing!! 🤝💪")
                        st.session_state.streak = 0
                        motivational_quotes = quiz_data.get("motivational_quotes", [])
                        if motivational_quotes:
                            st.info(random.choice(motivational_quotes))
                    st.session_state.answered_today = True
                    st.session_state.last_played = today
                    st.session_state.quiz_started = False

    # Reset Quiz Button (for development)
    if st.button("Reset Quiz (Dev Mode)", key="reset_quiz_button"): # Added unique key
        st.session_state.language = None
        st.session_state.difficulty = None
        st.session_state.questions = []
        st.session_state.user_answers = []
        st.session_state.score = 0
        st.session_state.answered_today = False
        st.session_state.quiz_started = False  # Ensure this is reset too
        st.rerun()
#===================== Job Search ============================= #
elif menu == "Job Search 💼":
    st.subheader("💼 Fetch Jobs")
    job_title = st.text_input("Enter job title:", placeholder = "eg., Cloud engineer")
    location = st.text_input("Enter location (optional): ", placeholder = "Pune")
    results_per_page = st.slider("Results per page: ", min_value = 5, max_value = 20, value = 10, step = 5)
    if st.button("Let's Fetch"):
        if job_title:
            search_url = f"https://api.adzuna.com/v1/api/jobs/gb/search/1?app_id={APP_ID}&app_key={APP_KEY}&what={quote_plus(job_title)}&&where={quote_plus(location)}&results_per_page={results_per_page}&content-type=application/json"
            try:
                response = requests.get(search_url)
                response.raise_for_status()
                data = response.json()
                if "results" in data and data["results"]:
                    st.subheader(f"Search results for '{job_title}' in '{location if location else 'any location'}'")
                    for job in data["results"]:
                        st.markdown(f"**{job['title']}**")
                        st.markdown(f"Company: {job['location']['display_name']}")
                        st.markdown(f"Location: {job['location']['display_name']}")
                        st.markdown(f"Salary: {job.get('salary_min', 'N/A')} - {job.get('salary_max', 'N/A')} {job.get('currency', '')}")
                        st.markdown(f"[Apply Now]({job['redirect_url']})")
                        st.divider()
                else:
                    st.info("No job listings found matching your criteria.")
            except requests.exceptions.RequestException as e:
                st.error(f"An error occurred during the API request: {e}")
                st.error(f"Details: {e}")
            except json.JSONDecodeError:
                st.error("Error decoding the API response.")
                st.error(f"Response content: {response.content if 'response' in locals() else 'No response received'}")
        else:
            st.warning("Please enter a job title to search.")

#---------------------------- Resume Analysis --------------------------#
elif menu == "Resume Analysis 📄":
    st.subheader("📄 Upload Your Resume for Analysis")
    uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"]) # Limiting to PDF for simplicity in this step
    print(f"Uploaded File: {uploaded_file}")  # Debugging line
    if uploaded_file is not None:
        resume_text = extract_text_from_pdf(uploaded_file)
        if resume_text:
            st.success("✅ Resume Text Extracted Successfully!")
            skills = extract_skills(resume_text)
            experience = detect_experience(resume_text)
            degree = detect_degree(resume_text)
            score = generate_score(skills, experience, degree)
            next_steps = suggest_next_steps(skills, experience)
            badges = award_badges(skills)
            # --- Display Analysis --- #
            st.header("📋 Resume Summary")
            st.subheader("🎯 Key Skills")
            st.write(", ".join(skills) if skills else "No major skills detected.")
            st.subheader("📈 Experience Level")
            st.write(f"{experience} years" if experience else "No experience mentioned.")
            st.subheader("🎓 Degree")
            st.write(degree)
            st.subheader("🔍 Formatting Tips")
            if any(keyword in resume_text.lower() for keyword in ["education", "skills", "experience", "contact"]):
                st.write("Good formatting detected ✅")
            else:
                st.warning("Consider adding sections like Education, Skills, Experience!")
            st.subheader("🏆 Overall Resume Score")
            st.progress(score)
            st.success(f"Your Resume Score: {score}/100")
            st.subheader("🚀 Suggested Next Steps")
            for tip in next_steps:
                st.write(f"- {tip}")
            st.subheader("🎖️ Badges Earned")
            if badges:
                for badge in badges:
                    st.balloons()
                    st.success(badge)
            else:
                st.info("No badges earned yet. Add more skills!")
            st.subheader("🌟 Skill Cloud")
            if skills:
                wordcloud = generate_wordcloud(skills)
                fig, ax = plt.subplots()
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
        else:
            st.error("⚠️ Could not extract any text from the uploaded PDF.")
