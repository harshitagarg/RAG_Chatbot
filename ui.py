import streamlit as st
import requests

st.title("🤖 RAG Chatbot")

if "chat" not in st.session_state:
    st.session_state.chat = []

query = st.chat_input("Ask your question")  # Using Streamlit's chat input for better UI

if query and query.strip():
    with st.spinner("Thinking..."):
        res = requests.post("http://localhost:8000/ask", json={"question": query})
        # try:
        #     response_data = res.json()
        #     answer = response_data.get("answer", "No answer provided")
        #     sources = response_data.get("source_documents", [])
        # except requests.exceptions.JSONDecodeError:
        #     st.error("Failed to decode JSON response from the server.")
        # st.write(f"Raw response: {res.text}")
        answer = res.json().get("answer", "No answer provided")
        sources = res.json().get("source_documents", [])       

        st.session_state.chat.append(("You", query, None))
        source_text = sources[0]["source"] if sources else None
        st.session_state.chat.append(("Bot", answer, source_text))

for role, msg, src in st.session_state.chat:
    st.write(f"**{role}:** {msg}")
    
    if role == "Bot" and src:
        st.caption(f"📄 Source: {src}")