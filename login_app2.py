import streamlit as st
import mysql.connector
import hashlib
import subprocess
from streamlit_option_menu import option_menu

def create_connection():
    return mysql.connector.connect(
        host="gondola.proxy.rlwy.net",
        port=40948,
        user="root",
        password="HVbWzknILwIQeJJHEDPqfGMAjeaycSKh",
        database="railway"
    )

# Hashing function for password security
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# User Authentication Functions
def signup_user(username, password):
    connection = create_connection()
    cursor = connection.cursor()
    # Check if the username already exists
    cursor.execute("SELECT * FROM users WHERE username=%s", (username,))
    if cursor.fetchone():
        st.error("Username already exists. Please choose a different username.")
        cursor.close()
        connection.close()
        return

    # Insert the new user
    cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hash_password(password)))
    connection.commit()
    cursor.close()
    connection.close()
    st.success("Account created successfully! Please log in.")

def login_user(username, password):
    connection = create_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM users WHERE username=%s AND password=%s", (username, hash_password(password)))
    user = cursor.fetchone()
    cursor.close()
    connection.close()
    return user is not None

# Display Sign Up and Login Pages
def login_signup():
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    with st.sidebar:
        choice = option_menu("Menu", ["Login", "Sign Up"], icons=['login', 'person-plus'])

    if choice == "Sign Up":
        st.title("Sign Up")
        new_user = st.text_input("Username")
        new_password = st.text_input("Password", type='password')
        re_enter_password = st.text_input("Re-enter Password", type='password')

        if new_password and len(new_password) < 8:
            st.error("Password must be at least 8 characters long.")

        if st.button("Create Account"):
            if new_password != re_enter_password:
                st.error("Passwords do not match.")
            elif len(new_password) >= 8:
                signup_user(new_user, new_password)
            else:
                st.error("Please ensure your password meets the requirements.")

    elif choice == "Login":
        st.title("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type='password')
        if st.button("Login"):
            if login_user(username, password):
                st.session_state['logged_in'] = True
                st.success(f"Welcome, {username}!")
                # Redirect to main application after successful login
                st.write("Redirecting to main application...")
                launch_main_app()
            else:
                st.error("Invalid Username or Password")

# Function to launch main application
def launch_main_app():
    # Launch the main_app.py script in a new subprocess
    subprocess.Popen(["streamlit", "run", "main6.py"])

def main():
    login_signup()

if __name__ == "__main__":
    main()
