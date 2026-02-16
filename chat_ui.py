# Receipt Vault - Chat with Data
import streamlit as st  # type: ignore
import pandas as pd  # type: ignore
from database.queries import fetch_all_receipts  # type: ignore
from ai.gemini_client import GeminiClient  # type: ignore

def render_chat():
    st.header("💬 Chat with your Receipts")
    st.info("Ask questions about your spending, vendors, or trends using natural language.")

    # 1. Fetch Data for Context
    receipts = fetch_all_receipts()
    if not receipts:
        st.warning("No data found. Please upload receipts first to enable chat.")
        return

    df = pd.DataFrame(receipts)
    
    # 2. Chat history initialization
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 3. Chat Input
    if prompt := st.chat_input("E.g., 'How much did I spend at Amazon last month?'"):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 4. Generate AI response
        api_key = st.session_state.get("GEMINI_API_KEY")
        if not api_key:
            with st.chat_message("assistant"):
                st.error("Please set your Gemini API Key in the sidebar to use Chat.")
            return

        with st.chat_message("assistant"):
            with st.spinner("Analyzing your data..."):
                try:
                    client = GeminiClient(api_key)
                    # Prepare data summary for context
                    summary = df.to_string(index=False)
                    response = client.chat_with_data(prompt, summary)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Chat failed: {e}")