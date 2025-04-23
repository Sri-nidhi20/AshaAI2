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

#--------------------------defining quiz data -----------------------------#
def load_quiz_data():
    with open("quiz_data.json", "r") as f:
        return json.load(f)
quiz_data = load_quiz_data()
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
# ------------------ UTILS ------------------ #
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
logging.basicConfig(filename = "ashaai_log.txt", level=logging.INFO)
def query_gemini(prompt_text, timeout_seconds=60):
    logging.info(f"[{timestamp}] User prompt: {prompt_text}")

    greetings = r"^(hello|hi|hey|greetings|good morning|good afternoon|good evening)\b.*"
    if re.match(greetings, prompt_text, re.IGNORECASE):
        return "Hello there! How can I assist you with your career journey today?"

    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(prompt_text)
    logging.info(f"[{timestamp}] Sentiment scores: {vs}")
    if vs['compound'] < -0.2:  # Adjust this threshold as needed
        encouragement_query = f"The user is expressing negative feelings like '{prompt_text}'. Offer a short, encouraging and supportive message related to career challenges. Keep it concise and uplifting."
        logging.info(f"[{timestamp}] Detected negative sentiment, sending encouragement query to Gemini-2.0-flash: {encouragement_query}")
        try:
            contents = [{"parts": [{"text": encouragement_query}]}]
            response = model.generate_content(contents)  # Removed generation_config
            if response.text:
                logging.info(f"[{timestamp}] Gemini-2.0-flash encouragement response (first 50 chars): {response.text[:50]}...")
                return response.text
            else:
                return "Sending you some positive vibes! Remember that career journeys have ups and downs. How can I help you navigate this?"
        except Exception as e:
            logging.error(f"[{timestamp}] Error fetching encouragement from Gemini-2.0-flash: {e}")
            return f"Error: {str(e)}"

    motivation_keywords = r"\b(motivate|inspire|inspiration|encouragement|uplift|positive outlook|give me motivation)\b"
    if re.search(motivation_keywords, prompt_text, re.IGNORECASE):
        motivation_query = f"Give me a short, inspiring message related to {prompt_text.lower().replace('give me motivation', '').strip()}. Keep it concise and uplifting."
        logging.info(f"[{timestamp}] Sending motivation query to Gemini-2.0-flash: {motivation_query}")
        try:
            contents = [{"parts": [{"text": motivation_query}]}]
            response = model.generate_content(contents)  # Removed generation_config
            if response.text:
                logging.info(f"[{timestamp}] Gemini-2.0-flash motivation response (first 50 chars): {response.text[:50]}...")
                return response.text
            else:
                return "Here's a little something to keep you going: Every challenge is an opportunity to learn and grow."
        except Exception as e:
            logging.error(f"[{timestamp}] Error fetching motivation from Gemini-2.0-flash: {e}")
            return "I'm experiencing a slight delay. Please try your request again."

    career_keywords = r"\b(B.Tech|BE|B.SC|BCA|MTECH|ME|MSC|MBA|PhD|IT|CS|ECE|EEE|ME|CE|Engineering|Biotechnology|data science|artificial inteliigence(AI)|Machine learning(ML)|cybersecurity|software engineering|business analytics|management studies|BCOM|MCOM|BA|MA|BDes|BPharm|BArch|software engineer|data analyst|data scientist|web developer|front-end developer|back-end developer|full-stack developer|mobile app developer(iOS, Android)| cloud engineer| DevOps engineer| cybersecurity analyst| network engineer|database administrator|project manager|business analyst|marketing specialist|sales representative|human resources (HR) generalist| technical support engineer| quality assurance(QA) tester|UI/UX designer|Product Manager|Research ScientistManagement Consultant|Financial Analyst|Accountant|Operations Manager|Chief Technology Officer (CTO)|Chief Executive Officer (CEO)|Team Lead|Architect (Software, Solutions, Enterprise)|Specialist (in various domains)|Associate|Analyst|Engineer|Developer|Consultant|Manager|Director|VP (Vice President)|Programming Languages (Python, Java, C++, JavaScript, C#, Go, etc.)|Data Analysis Tools (Pandas, NumPy, SQL, R)|Machine Learning Algorithms (Regression, Classification, Clustering, Deep Learning)|Cloud Platforms (AWS, Azure, GCP)|DevOps Tools (Docker, Kubernetes, Jenkins, Git)|Cybersecurity Concepts (Network Security, Cryptography, Ethical Hacking)|Database Management (SQL, NoSQL)|Web Development Frameworks (React, Angular, Vue.js, Node.js, Django, Flask)|Mobile Development (Swift, Kotlin, Flutter, React Native)|Testing Frameworks (JUnit, Selenium, Cypress)|Operating Systems (Linux, Windows)|Networking Concepts (TCP/IP, DNS, Routing)|Big Data Technologies (Spark, Hadoop)|UI/UX Design Tools (Figma, Sketch, Adobe XD)|Data Visualization (Tableau, Power BI)|Communication (Written and Verbal)|Problem-Solving|Critical Thinking|Teamwork|Collaboration|Leadership|Time Management|Adaptability|Learning Agility|Interpersonal Skills|Presentation Skills|Negotiation|Creativity|Emotional Intelligence|Placement|Recruitment|Hiring|Internship|Training|Career Fair|Job Portal|Application|Interview (Technical, HR, Behavioral)|Resume|Curriculum Vitae (CV)|Cover Letter|Networking|LinkedIn|Portfolio|Personal Branding|Skill Development|Upskilling|Reskilling|Career Path|Job Market|Industry Trends|Company Culture|Compensation|Benefits|Growth Opportunities|Professional Development|Alumni Network|Placement Cell|Company|Job Description|Eligibility Criteria)\b"
    if re.search(career_keywords, prompt_text, re.IGNORECASE) or "career" in prompt_text.lower() or "job" in prompt_text.lower():
        logging.info(f"[{timestamp}] Assuming career-related query, sending to Gemini-2.0-flash: {prompt_text}")
        try:
            contents = [{"parts": [{"text": prompt_text}]}]
            response = model.generate_content(contents)  # Removed generation_config
            if response.text:
                logging.info(f"[{timestamp}] Gemini-2.0-flash response (first 50 chars): {response.text[:50]}...")
                return response.text
            else:
                return "Hmm, I didn't get a clear response for that career query. Could you please rephrase?"
        except Exception as e:
            logging.error(f"[{timestamp}] Error in query_gemini (Gemini-2.0-flash) for career query: {e}")
            return "Sorry! I'm having trouble processing your request right now. Please try again in few moments."
    else:
        return "I'm designed to be a helpful companion for your career journey. While I appreciate your message, I'm best equipped to answer questions related to careers, job opportunities, professional development, and provide encouragement. How can I specifically help you with your career today?"

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

    for i, (sender, msg) in enumerate(st.session_state.chat):
        if sender == "user":
            st.markdown(f"👩‍💼 You:** {msg}")
        else:
            st.markdown(f"👩 AshaAI:** {msg}")
    user_input = st.chat_input("Your Question...")
    if user_input:
        st.markdown(f"👩‍💼 You:** {user_input}")
        st.session_state.chat.append(("user", user_input))
        st.session_state.pending_input = user_input
        st.session_state.chat_turn += 1

    if st.session_state.pending_input:
        prompt_text = st.session_state.pending_input
        with st.spinner("AshaAI is thinking..."):
            reply = query_gemini(prompt_text)
        placeholder = st.empty()
        typed_response = ""
        for char in reply:
            typed_response += char
            placeholder.markdown(f"👩 AshaAI:** {typed_response}")
            time.sleep(0.01)
        st.session_state.chat.append(("AshaAI", reply))
        st.session_state.pending_input = None

# ------------------ CHAT HISTORY ------------------ #
elif menu == "Chat History 🗨":
    st.subheader("📜 Chat History")

    if "chat_history" not in st.session_state:
        try:
            with open(history_file, "r") as f:
                st.session_state.chat_history = json.load(f)
        except FileNotFoundError:
            st.session_state.chat_history = []

    if "current_chat_saved" not in st.session_state:
        st.session_state.current_chat_saved = False

    if "chat" in st.session_state and st.session_state.chat and not st.session_state.current_chat_saved:
        st.session_state.chat_history.append(st.session_state.chat.copy())
        st.session_state.chat = []
        st.session_state.current_chat_saved = True
        try:
            with open(history_file, "w") as f:
                json.dump(st.session_state.chat_history, f)
        except Exception as e:
            st.error(f"Error saving chat history: {e}")

    if not st.session_state.chat_history:
        st.info("No previous chats available.")
    else:
        chat_titles = [f"Chat {i + 1}" for i in range(len(st.session_state.chat_history))]
        selected_chat_title = st.radio("Select a previous chat to view:", chat_titles)

        if selected_chat_title:
            selected_chat_index = chat_titles.index(selected_chat_title)
            selected_chat = st.session_state.chat_history[selected_chat_index]
            st.subheader(f"Content of {selected_chat_title}:")
            for role, content in selected_chat:
                st.markdown(f"{'👩‍💼 You:' if role == 'user' else '👩 AshaAI:'}** {content}")

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
    st.subheader("🔍 Search Your Chats (Coming Soon)")

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
        st.selectbox("Choose a programming language:", list(quiz_data.keys()), key="language")

        # Difficulty Selection (only if language is selected)
        if st.session_state.language:
            st.selectbox("Select difficulty level:", ["Easy", "Medium", "Hard"], key="difficulty")

        # Start Quiz Button (appears after language and difficulty are selected)
        if st.session_state.language and st.session_state.difficulty:
            if st.button("Start Quiz"):
                # Load questions if not already loaded
                if not st.session_state.questions:
                    all_questions = quiz_data[st.session_state.language][st.session_state.difficulty.lower()]
                    st.session_state.questions = random.sample(all_questions, 3)
                    st.session_state.user_answers = [""] * 3  # Initialize as a list of empty strings
                    st.session_state.score = 0
                    st.session_state.quiz_started = True  # Flag to indicate quiz has started
                    st.session_state.answered_today = False  # Reset for fresh quiz day
                    st.rerun()  # Force a rerun to display questions

        # Allow switching between questions or languages after quiz is started
        if st.session_state.get('quiz_started'):
            for i, q in enumerate(st.session_state.questions):
                st.markdown(f"**Question {i+1}:** {q['question']}")
                st.session_state.user_answers[i] = st.text_input(
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
                        st.markdown(f"**Q{i+1}:** {st.session_state.questions[i]['question']}")
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
                    st.session_state.quiz_started = False  # Reset quiz started flag

    # Reset Quiz Button (for development)
    if st.button("Reset Quiz (Dev Mode)"):
        st.session_state.language = None
        st.session_state.difficulty = None
        st.session_state.questions = []
        st.session_state.user_answers = []
        st.session_state.score = 0
        st.session_state.answered_today = False
        st.session_state.quiz_started = False  # Ensure this is reset too
        st.rerun()
