import streamlit as st
import requests
import argostranslate.package
import argostranslate.translate

# Ollama API details
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"

# Language mapping
LANGUAGE_CODES = {
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta"
}

# Load translation package if needed
def install_translation_package(from_code, to_code):
    available_packages = argostranslate.package.get_available_packages()
    for pkg in available_packages:
        if pkg.from_code == from_code and pkg.to_code == to_code:
            path = pkg.download()
            argostranslate.package.install_from_path(path)
            break

# Translate text offline using Argos
def translate_text(text, from_lang="en", to_lang="hi"):
    if from_lang == to_lang:
        return text
    install_translation_package(from_lang, to_lang)
    return argostranslate.translate.translate(text, from_lang, to_lang)

# Call local LLM via Ollama API
def query_ollama(prompt):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        return response.json().get("response", "No response.")
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit UI
st.set_page_config(page_title="NyayaGPT", layout="centered")
st.title("⚖️ NyayaGPT")
st.write("Get your legal doubts solved (Indian Constitution)")

# Language selection
language = st.selectbox("Select Output Language:", ["English", "Hindi", "Tamil"])

# Prompt input
user_input = st.text_area("Enter your prompt", height=200)

# Submit
if st.button("Submit"):
    if user_input.strip():
        user_input_prefixed = "Answer the questions according to Indian Constitution only:  " + user_input
        with st.spinner("Generating response..."):
            english_response = query_ollama(user_input_prefixed)
            target_lang_code = LANGUAGE_CODES[language]
            translated = translate_text(english_response, from_lang="en", to_lang=target_lang_code)
            st.success(f"Response in {language}:")
            st.write(translated)
    else:
        st.warning("Please enter a prompt first.")
