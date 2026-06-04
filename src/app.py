import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from generate import answer

st.set_page_config(page_title="RAG Smart Doc Chatbot", page_icon="📄", layout="centered")

st.title("📄 RAG Smart Doc Chatbot")
st.caption("Ask questions about your documents. Answers are grounded in your files with citations.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📚 Sources used"):
                for s in msg["sources"]:
                    page_info = f", page {s['page']}" if s['page'] != "" else ""
                    st.markdown(f"**{os.path.basename(s['file'])}{page_info}**")
                    st.caption(s["excerpt"] + "...")

if prompt := st.chat_input("Ask something about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            result = answer(prompt)

        st.markdown(result["answer"])

        if result["sources"]:
            with st.expander("📚 Sources used"):
                for s in result["sources"]:
                    page_info = f", page {s['page']}" if s['page'] != "" else ""
                    st.markdown(f"**{os.path.basename(s['file'])}{page_info}**")
                    st.caption(s["excerpt"] + "...")
        else:
            st.warning("No relevant sources found in your documents.")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })
