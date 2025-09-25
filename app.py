import streamlit as st
import pickle
import faiss
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

# Load FAISS index + metadata
with open("vectorstore.pkl", "rb") as f:
    store = pickle.load(f)

index = store["index"]
metadata = store["metadata"]

st.set_page_config(
    page_title="DPA 2019",
    page_icon="⚖️"
)

client = Groq(api_key=st.secrets.Keys.GROQ_API_KEY)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello 👋 I’m your DPA chatbot. Ask me anything about the Data Protection Act 2019"}
    ]
custom_replies = {
    "hello": "Hello 👋! I’m your DPA assistant. Ask me anything about the Data Protection Act, compliance, or data privacy.",
    "hi": "Hi there 👋, how can I help you with data protection today?",
    "hey": "Hey! 👋 Ready to dive into data privacy questions?",
    "who are you": "I’m your DPA chatbot 🤖, trained on the Data Protection Act and related materials.",
    "what is dpa": "DPA stands for Data Protection Act 📘. It’s the law that governs how personal data should be handled securely and fairly.",
    "good morning": "Good morning 🌞! Let’s talk data protection.",
    "good evening": "Good evening 🌙! What would you like to know about the DPA?",
    "what files have you trained with": " Data protection course files"
}
st.title("Want to get quick answers on DPA 2019?")

# Sidebar controls
with st.sidebar:

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask me anything about the Data Protection Act 2019"}
        ]
        st.rerun()
    st.image("dpa.png", use_container_width=True)
    st.markdown("🎓 Want to learn more? [**Enroll for the course here!**](https://www.linkedin.com/company/centre-for-intellectual-property-and-information-technology-law/)")
# Preset questions
st.subheader("💡 Try one of these questions:")
preset_qs = [
    "What are the key principles of data protection?",
    "When is a Data Protection Impact Assessment required?",
    "How should organizations respond to a data breach?"
]
cols = st.columns(len(preset_qs))
preset_clicked = None
for i, q in enumerate(preset_qs):
    if cols[i].button(q):
        preset_clicked = q

# Display past chat history with two-column layout
for msg in st.session_state.messages:
    if msg["role"] == "user":
        col1, col2 = st.columns([2, 3])
        with col2:
            st.markdown(
                f"<div style='background-color:#f0f0f0; padding:10px; border-radius:10px;'>{msg['content']}</div>",
                unsafe_allow_html=True
            )
        with col1:
            st.empty()
    else:
        col1, col2 = st.columns([2, 3])
        with col2:
            st.empty()
        with col1:
            st.markdown(
                f"<div style='background-color:#d1e7ff; padding:10px; border-radius:10px;'>{msg['content']}</div>",
                unsafe_allow_html=True
            )

# Input box (always visible)
user_input = st.chat_input("Type your question...")

# Decide which query to process
query = preset_clicked or user_input

def search_vectorstore(query, k=3):
    """Retrieve top-k relevant chunks from vectorstore."""
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, idxs = index.search(q_emb.astype("float32"), k)
    return [metadata[i]["text"] for i in idxs[0]]

if query:
    # Add user query
    st.session_state.messages.append({"role": "user", "content": query})

    col1, col2 = st.columns([2, 3])
    with col2:
        st.markdown(
            f"<div style='background-color:#f0f0f0; padding:10px; border-radius:10px;'>{query}</div>",
            unsafe_allow_html=True
        )
    with col1:
        st.empty()

    key = query.lower().strip()
    if key in custom_replies:
        answer = custom_replies[key]
    else:

        # Retrieve context
        context = "\n\n".join(search_vectorstore(query))

        # Build prompt
        prompt = f"""You are a helpful expert on data protection law. 
        Use the following context to answer the question concisely. 
        If the answer is not in the context, say you don’t know.

        Context:
        {context}

        Question: {query}
        Answer:"""

        # Call Groq API
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # 
            messages=[
                {"role": "system", "content": "You are a DPA expert."},
                {"role": "user", "content": prompt}
            ]
        )
        answer = response.choices[0].message.content

    # Add assistant reply
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Display bot reply immediately on right
    col1, col2 = st.columns([2, 3])
    with col2:
        st.empty()
    with col1:
        st.markdown(
            f"<div style='background-color:#d1e7ff; padding:10px; border-radius:10px;'>{answer}</div>",
            unsafe_allow_html=True
        )

    # Sources
    with st.expander("ℹ️ Sources"):
        try:
            for text in context.split("\n\n"):
                st.write("-", text[:300], "...")
        except:
            print()


