import streamlit as st
from PIL import Image
import google.generativeai as genai

# Configure Google API key directly
API_KEY = "** YOUR_APP_KEY **"
genai.configure(api_key=API_KEY)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

st.set_page_config(page_title="AI Image Captioning", page_icon="🧑‍💻", layout="wide")

# Function to get Gemini chat response
def get_gemini_response(question):
    model = genai.GenerativeModel("gemini-pro")
    chat = model.start_chat(history=[])
    response = chat.send_message(question, stream=True)
    return response

# Streamlit interface
try:
    st.title("AI Image Captioning ")

    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox("Go to", ["Home", "ChatBot", "Image Captioning"])

    if page == "Home":
        st.header("Welcome to the AI Tools Application")
        st.write(""" Image captioning is the process of generating a textual description or caption that accurately and concisely describes the content and context of a given image. This pertains to the unification of the approaches of computer vision and natural language processing in order to analyse visual data. 
        
        Image captioning typically involves the following steps:
         
        Image Feature Extraction: Features such as shapes, colors, textures, and objects are extracted from the image.
        Language Modeling: A language model is used to generate a sequence of words that describe the image features.
        Caption Generation: The generated words are assembled into a coherent and grammatically correct caption.
        """)
        images = ["img1.jpeg", "img2.jpeg", "img3.jpeg"]
        cols = st.columns(3)
    
        for idx, image_path in enumerate(images):
            with cols[idx % 3]:
                st.image(image_path, use_container_width=True)
    
    elif page == "ChatBot":
        st.title("ChatBot Service")
        user_input = st.text_input("Input:", key="input")
        submit = st.button("Ask the Question")
        if submit and user_input:
            response = get_gemini_response(user_input)
            st.session_state['chat_history'].append(("You", user_input))
            st.subheader("Response")
            for chunk in response:
                st.write(chunk.text)
                st.session_state['chat_history'].append(("Bot", chunk.text))

        st.subheader("Chat History")
        for role, text in st.session_state['chat_history']:
            st.write(f"{role}: {text}")

    elif page == "Image Captioning":
        st.title("Generate Caption with Hashtags")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None and st.button('Upload'):
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
                img = Image.open(uploaded_file)
                caption = model.generate_content([
                    "Generate a detailed caption that accurately describes the content, mood, and potential story of the image in English.",
                    img
                ])
                tags = model.generate_content([
                    "Generate 10 trending hashtags for the image in a line in English.",
                    img
                ])
                st.image(img, caption=f"Caption: {caption.text}")
                st.write(f"Tags: {tags.text}")
            except Exception as e:
                st.error(f"Failed to generate caption due to: {str(e)}")

except Exception:
    st.error("OOPS! SOMETHING WENT WRONG.")
