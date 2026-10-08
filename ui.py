"""
Minimal chat UI (Streamlit).

    streamlit run ui.py

Deliberately plain: no custom CSS, no theme, no colour - just the answer,
the confidence score and the retrieved chunks.
"""

import streamlit as st

import rag

st.set_page_config(page_title="Agentic AI RAG Chatbot", layout="centered")

st.title("Agentic AI RAG Chatbot")
st.caption("Answers are restricted to the 'Agentic AI' ebook (Konverge AI).")

question = None
submitted = False
with st.form("ask_form"):
    question = st.text_input("Question", placeholder="What is Agentic AI?")
    submitted = st.form_submit_button("Ask")

if submitted:
    if not question.strip():
        st.write("Please type a question.")
    else:
        with st.spinner("Retrieving chunks and generating an answer..."):
            result = rag.answer_question(question.strip())

        st.subheader("Answer")
        st.write(result["answer"])

        st.subheader("Confidence")
        st.write(f"{result['confidence']:.3f}")

        st.subheader("Retrieved context chunks")
        if not result["contexts"]:
            st.write("No chunks were retrieved.")
        for i, chunk in enumerate(result["contexts"], 1):
            with st.expander(f"Excerpt {i} - page {chunk['page']} - score {chunk['score']:.3f}"):
                st.write(chunk["text"])
