import streamlit as st
import google.generativeai as genai
import os
import requests
import pandas as pd
from datetime import datetime
timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
from PIL import Image
import time
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
quiz_data = {
    "C": {
        "easy": [
            {"question": "What is the difference between #include <stdio.h> and #include 'stdio.h'?"},
            {"question": "Write a program to check if a number is odd or even."},
            {"question": "Define a pointer and give an example."},
            {"question": "What is the size of int, float, char and double in C ?"},
            {"question": "Explain the purpose of scanf() and printf()."},
            {"Question": "What is a function in C ?"},
             ],
        "medium": [
            {"question": "Write a function to find the factorial of a number using recursion."},
            {"question": "Explain the difference between a structure and a union."},
            {"question": "Implement a program to reverse a string without using a library function."},
            {"question": "What are storage classes in C ?"},
            {"question": "Explain stack and heap memory."},
            {"question": "What is a segmentation fault?"},
        ],
        "hard": [
            {"question": "Implement a stack using an array."},
            {"question": "Explain memory leaks and how to avoid them in C ?"},
            {"question": "Write a program to sort an array using quicksort."},
            {"question": "What is a pointer arithmetic?"},
            {"question": "How would you simulate object-oriented programming in C?"},
            {"question": "Explain the working of function pointers."},
        ],
    },
    "C++": {
        "easy": [
            {"question": "What is the difference between C and C++?"},
            {"question": "Define a class and an object with a simple example."},
            {"question": "Explain the concept of function overloading."},
            {"question": "What is a constructor and destructor?"},
            {"question": "What is the use of the "this" pointer?"},
            {"question": "Write a program to dispplay "Hello, World!" using a class."},
        ],
        "medium": [
            {"question": "Implement a class with getter and setter methods."},
            {"question": "Explain inheritance with a practical example."},
            {"question": "What are templates in C++?"},
            {"question": "What is operator overloading? Give an example."},
            {"question": "How is memory managed in C++?"},
            {"question": "What are the differences between deep copy and shallow copy?"},
        ],
        "hard": [
            {"question": "Implement a linked list using classes."},
            {"question": "Explain virtual functions and pure virtual functions."},
            {"question": "How does polymorphism work in C++?"},
            {"question": "What is the role of std::move and rvalue refernces?"},
            {"question": "Implement a basic LRU Cache using STL."},
            {"question": "Discuss RAII (Resource Acquisition Is Initialization)."},
        ],
    },
    "Java": {
        "easy": [
            {"question": "What is the JVM and how is it different from JRE and JDK?"},
            {"question": "What is the difference between == and .equals()?"},
            {"question": "Write a Java program to check if a number is prime."},
            {"question": "Define a class and create an object in Java."},
            {"question": "What is the purpose of the main() method?"},
            {"question": "What are primitive data types in Java?"},
        ],
        "medium": [
            {"question": "Explain the concept of inheritance with an example."},
            {"question": "Implement a simple class hierarchy for animals."},
            {"question": "What are the differences between an interface and an abstract class?"},
            {"question": "What is exception handling in Java?"},
            {"question": "Write a method to reverse a string using a StringBuilder."},
            {"question": "How does garbage collection work in Java?"},
        ],
        "hard": [
            {"question": "Implement a custom exception in Java."},
            {"question": "Write a program to implement multithreading using Runnable."},
            {"question": "How does the JVM optimize bytecode execution?"},
            {"question": "Implement a Singleton design pattern."},
            {"question": "What is the pupose of the volatile keyword?"},
            {"question": "Implement a producer-consumer problem using threads."},
        ],
    },
    "Go(Golang)": {
        "easy": [
            {"question": "What is a goroutine in GO?"},
            {"question": "Write a Go program to print "Hello, World!"."},
            {"question": "How do you declare and initialize a slice?"},
            {"question": "Explain the use of := in variable declarations."},
            {"question": "Write a function to check if a number is even or odd."},
            {"question": "What is the differnce between an array and a slice?"},
        ],
        "medium": [
            {"question": "Write a program to use goroutines and channels to print concurrently."},
            {"question": "What is the use of defer in Go?"},
            {"question": "Implement a map that counts character frequencies in a string."},
            {"question": "Explain how Go hnadles errors without try-catch."},
            {"question": "What is the select statement used for?"},
            {"question": "How is memory managed in Go?"},
        ],
        "hard": [
            {"question": "Implement a worker pool pattern using goroutines."},
            {"question": "Optimize a program that concurrently fetches data from APIs."},
            {"question": "How do you avoid race conditions in Go?"},
            {"question": "Write a basic REST API using Go's standard library."},
            {"question": "Implement a custom data structure (eg., queue) in Go."},
            {"question": "Discuss garbage collection and concurrency safety in Go."},
             ],
    },
    "Rust": {
        "easy": [
            {"question": "What makes Rust a memory-safe language?"},
            {"question": "Write a Rust program to print the Fibonacci Series."},
            {"question": "What is borrow in Rust?"},
            {"question": "Explain the differnce between String and &str."},
            {"question": "Write a function that adds two numbers and returns the result."},
            {"question": "What is pattern matching and how is match used?"},
        ],
        "medium": [
            {"question": "Implement a program that reads from a file and counts words."}
            {"question": "Explain ownership, borrowing, and lifetimes."},
            {"question": "What is Option<T> and how do you handle it ?"},
            {"question": "Write a Rust program that handles command line arguments."},
            {"question": "How are traits different from interfaces in other languages?"},
            {"question": "Implement a struct with associated methods and derive traits."},
        ],
        "hard": [
            {"question": "Build a multi-threaded program using channels."},
            {"question": "Optimize file I/O using buffered reading and error handling."},
            {"question": "Implement a custom smart pointer."},
            {"question": "Write a simple web server using hyper and actix-web."},
            {"question": "How does Rust handle memory without garbage collection?"},
            {"question": "Use lifetimes to manage  refernces across function boundaries."},
        ],
    },
    "C#": {
        "easy": [
            {"question": "What is the CLR in C#?"},
            {"question": "Write a C# method to find the square of a number."},
            {"question": "What is the differnce between ref and out?"},
            {"question": "What is the namespace in C#?"},
            {"question": "How do you create a class and instantiate an object?"},
            {"question": "What are value types and refernce types?"},
        ],
        "medium": [
            {"question": "Implement a class with properties and use getters/setters."},
            {"question": "What is a delegate and how is it differnt from an event?"},
            {"question": "Explain boxing and unboxing with examples."},
            {"question": "Write a program to filter even numbers from alist using LINQ."},
            {"question": "How is exception handling done in C#?"},
            {"question": "What are static classes and when should they be used?"},
        ],
        "hard": [
            {"question": "Implement a multi-threaded application using Task."},
            {"question": "What are async and await in asynchronous programming?"},
            {"question": "How would you implement dependency injection in C#?"},
            {"question": "Design a mini inventory system using OOP in C#."},
            {"question": "Explain how garbage collection works in .NET."},
            {"question": "Create a custom collection that implements IEnumerable."},
        ],
    },
    "Python": {
        "easy": [
            {"question": "What are Python's key data types?"},
            {"question": "Write a Python program to check if a string is a palindrome."},
            {"question": "What's the difference between list and tuple?"},
            {"question": "What is indentation and why is it important?"},
            {"question": "Write a function to count vowels in a string."},
            {"question": "What is the use of if__name__ == "__main__" "},
        ],
        "medium": [
            {"question": "Write a function to find duplicate elements in alist."},
            {"question": "What is a lambda function? Give an example."},
            {"question": "Explain the differnce between deepcopy() and copy()."},
            {"question": "Implement a dictionary-based frequency counter."},
            {"question": "What are decorators? Write a sample."},
            {"question": "When would you use a generator instead of a list?"},
        ],
        "hard": [
            {"question": "Implement a class with inheritance and method overriding."},
            {"question": "Optimize code that parses a huge log file and extracts data."},
            {"question": "How does Python manage memory"},
            {"question": "Write a mini project like a to-do list CLI app."},
            {"question": "What are metaclasses? When are they useful?"},
            {"question": "Create a multi-threaded app using threading and queue."},
        ],
    },
    "Python for Data science/Analysis": {
        "easy": [
            {"question": "What are pandas, numpy and matplotlib used for?"},
            {"question": "How do you load CSV file using pandas?"},
            {"question": "Write a code to calculate the mean of a numeric column."},
            {"question": "What is the shape of a DataFrame?"},
            {"question": "How do you handle missing values in pandas?"},
            {"question": "How do you filter rows based on a condition?"},
        ],
        "medium": [
            {"question": "Create a histogram for a dataset column using matplotlib."},
            {"question": "Explain the difference between .loc[] and .iloc[]."},
            {"question": "Write a function to normalize a numeric column."},
            {"question": "How do you detect and remove duplicates?"},
            {"question": "What is the role of groupby() in pandas?"},
            {"question": "How do you merge two DataFrames?"},
        ],
        "hard": [
            {"question": "Perform EDA on a dataset and summarize findings."},
            {"question": "Optimize a data pipeline for real-time analytics."},
            {"question": "Implement a regression model using scikit-learn."},
            {"question": "create a dashboard using Plotly or Streamlit."},
            {"question": "How would you handle a dataset with 1M+ rows efficiently?"},
            {"questions": "How do you identify and treat outliers?"},
        ],
    },
    "R": {
        "easy": [
            {"question": "What is a vector in R?"},
            {"question": "Write a function in R to find the average of a list."},
            {"question": "How do you create a DataFrame in R?"},
            {"question": "What are some common data types in R?"},
            {"question": "How to install and load a package in R?"},
            {"question": "What is the difference between = and <- in R?"},
        ],
        "medium": [
            {"question": "Use ggplot2 to plot a bar graph."},
            {"question": "How do you handle NA values in R?"},
            {"question": "What is the difference between apply(), lapply(), and sapply()?"},
            {"question": "Write a function that normalized a vector."},
            {"question": "How do you join two datasets in R?"},
            {"question": "Explain the use of dplyr package."},
        ],
        "hard": [
            {"question": "Perform a time series analysis using forecast."},
            {"question": "Build and evaluate a linear regression model."},
            {"question": "Write a custom function to compute correlation matrix with plots."},
            {"question": "Optimize a slow loop using vectorization."},
            {"question": "Build a Shiny app that visualized a dataset."},
            {"question": "Conduct principal component analysis (PCA) on a dataset."},
        ],
    },
    "Julia": {
        "easy": [
            {"question": "What is Julia used for?"},
            {"question": "How do you define a function in Julia?"},
            {"question": "Write a Julia program to check if a number is even."},
            {"question": "How is Julia different from Python?"},
            {"question": "What are tuples in Julia?"},
            {"question": "how do you install a package?"},
        ],
        "medium": [
            {"question": "Write a program to compute the dot product of two vectors."},
            {"question": "Explain type declarations and why they matter."},
            {"question": "How do you handle missing data in Julia?"},
            {"question": "What is broadcasting and how is it used?"},
            {"question": "Create a plot using the Plots.jl package."},
            {"question": "Write a function that performs linear regression."},
        ],
        "hard": [
            {"question": "Perform matrix operations on large datasets."},
            {"question": "Optimize a loop using @inbounds and @simd."},
            {"question": "Use Julia's multithreading to process a large file."},
            {"question": "Compare the performance of a function in Python and Julia."},
            {"question": "Build a basic statistical model using GLM.jl."},
            {"question": "Implement a caching system in Julia for faster computation."},
        ],
    },
}
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
# --------------------- QUIZ ---------------------------#
elif menu == "QUIZ TIME 🤩🥳":
    st.header("It's the Quiz Time!!")
    st.subheader("🎯 Ready, Set, Code! 💻 Time to show off your skills and conquer this quiz like a coding pro! 💥")
    today = date.today()
    if st.session_state['quiz_played_today'] and st.session_state['last_played_date'] == today:
        st.warning("You've already played the QUIZ today. Please comeback tomorrow to play again! Till the practice and stay tuned..🤗😉")
        return
    languages = ["C", "C++", "Java", "Go(Golang)", "Rust", "C#", "Python", "Python (for data analysis/science", "R", "Julia", "MATLAB", "HTML & CSS", "JavaScript", "TypeScript", "GraphQL", "Kotlin", "Swift", "Dart", "SQL", "PL/SQL", "T-SQL", "Bash/Shell", "Hashkell/Elixir"]
    selected_language = st.selectbox("Select a Programming Language:", languages)
    difficulties = ["easy", "Medium", "Hard"]
    selected_difficulty = st.selectbox("Select Difficulty Level:", difficulties, disabled=not selected_language)

    if selected_language and selected_difficulty and not st.session_state['quiz_questions']:
        st.session_state['quiz_language'] = selected_language
        st.session_state['quiz_difficulty'] = selected_difficulty
        st.session_state['quiz_questions'] = generate_quiz_questions(selected_language, selected_difficulty)
        st.session_state['question_index'] = 0
        st.session_state['user_answers'] = []
        st.session_state['correct_answers_count'] = 0

    if st.session_state['quiz_questions'] and st.session_state['question_index'] < len(st.session_state['quiz_questions']):
        question_data = st.session_state['quiz_questions'][st.session_state['question_index']]
        st.write(f"**Question {st.session_state['question_index'] + 1} ({st.session_state['quiz_difficulty']}):** {question_data['question']}")
        user_answer = st.text_input("Your answer:")
        if st.button("Submit"):
            st.session_state['user_answers'].append(user_answer)
            if evaluate_answer(user_answer, question_data['answer']):
                st.success("Correct!")
                st.session_state['correct_answers_count'] += 1
            else:
                st.error(f"Incorrect. The correct answer was: {question_data['answer']}")
            st.session_state['question_index'] += 1
    elif st.session_state['quiz_questions'] and st.session_state['question_index'] == len(st.session_state['quiz_questions']):
        st.subheader("Quiz Finished!")
        if st.session_state['correct_answers_count'] == len(st.session_state['quiz_questions']):
            st.balloons()
            st.success("Congratulations! You answered all questions correctly!")
            st.session_state['streak'] += 1
        else:
            st.error("OOPS!! Not all answers were correct.")
            st.info(f"You got {st.session_state['correct_answers_count']} out of {len(st.session_state['quiz_questions'])} correct.")
            st.info(f"Motivational Quote: {random.choice(st.session_state['motivational_quotes'])}")
            st.info("Focus, Practice and come back tomorrow..")
            st.session_state['streak'] = 0
        st.info(f"Your current streak: {st.session_state['streak']}")
        st.session_state['quiz_played_today'] = True
        st.session_state['last_played_date'] = today
        st.session_state['quiz_questions'] = []
