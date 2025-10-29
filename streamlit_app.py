import streamlit as st
import time
import base64
from src.app import RAGAssistant

# Page Configuration
st.set_page_config(
    page_title="JapaPolicy AI",
    page_icon="air.png",
    layout="centered",
    initial_sidebar_state="auto"
)

def get_image_base64(image_path):
    """Converts a local image to a Base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Get the Base64 string for your image
image_base64 = get_image_base64("UK.png")

st.markdown(
    f"""
    <div style="display: flex; align-items: center; justify-content: center; gap: 10px; text-align: center;">
        <img src="data:image/png;base64,{image_base64}" width="40">
        <h1>JapaPolicy AI</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "Your intelligent assistant for UK immigration rules. "
    "Powered by Google Gemini and advanced RAG."
)

# Caching the RAG Assistant
# Use st.cache_resource for objects that should persist across sessions and reruns
@st.cache_resource
def initialize_assistant():
    """
    Loads the RAGAssistant which connects to the pre-built vector database.
    This function is cached to run only once per server start/code change.
    """
    print("--- Streamlit Cache: Initializing RAG Assistant (will run once) ---")
    try:
        assistant = RAGAssistant()
        print("--- Streamlit Cache: RAG Assistant Ready ---")
        return assistant

    except Exception as e:
        # Catch errors during initialization (e.g., ChromaDB connection, model load)
        print(f"❌ FATAL ERROR during Streamlit assistant initialization: {e}")
        # Display error prominently in the Streamlit app UI
        st.error(
             "Fatal Error: Failed to initialize the AI Assistant. "
             f"Please ensure 'build_db.py' has run successfully and check terminal logs. Error details: {e}"
        )
        return None

# Attempt to get the initialized assistant from the cache
assistant = initialize_assistant()

# Only proceed if the assistant initialized successfully
if assistant:
    # st.session_state persists across reruns for a single user session
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Welcome to JapaPolicy AI! "
                    "How can I help you with your UK immigration questions today?"
                ),
                "sources": [],
                "metrics": {}
            }
        ]

    # Display Prior Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Only show expander for assistant messages that have details
            if message["role"] == "assistant" and (message.get("sources") or message.get("metrics")):
                with st.expander("Show Details"):
                    metrics = message.get("metrics", {})
                    if metrics:
                        emoji = metrics.get('emoji', '📊')
                        confidence = metrics.get('confidence', 'N/A').upper()
                        search_type = metrics.get('search_type', 'N/A')
                        top_sim = metrics.get('top_sim', 0) * 100
                        avg_sim = metrics.get('avg_sim', 0) * 100
                        chunks_retrieved = metrics.get('chunks', 'N/A')

                        st.markdown(f"**{emoji} Confidence: {confidence}**")
                        st.markdown(
                            f"**Metrics:** `Search: {search_type}` | "
                            f"`Chunks: {chunks_retrieved}` | "
                            f"`Top Match: {top_sim:.1f}%` | "
                            f"`Avg Match: {avg_sim:.1f}%`"
                        )

                    sources = message.get("sources", [])
                    if sources:
                        st.markdown("**📚 Sources:**")
                        for s in sources:
                            # Use .get() with default values for safety
                            st.markdown(
                                f"   - **{s.get('file','?')}** (Page {s.get('page','?')}) | "
                                f"*Similarity: {s.get('similarity',0):.1%} ({s.get('quality','?')})*"
                            )
                    
    # Handle New User Input
    if prompt := st.chat_input("Ask about UK immigration rules..."):
        # Append user message to session state
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            # Use a spinner during generation
            with st.spinner("🔍 Searching documents and thinking..."):
                try:
                    # Call the invoke method from the initialized assistant
                    result = assistant.invoke(prompt)

                    # Prepare metrics dict for storage and display
                    assistant_metrics = {
                        "emoji": result.get('confidence_emoji', '📊'),
                        "confidence": result.get('confidence', 'low'),
                        "search_type": result.get('search_type', 'N/A'),
                        "top_sim": result.get('top_similarity', 0),
                        "avg_sim": result.get('avg_similarity', 0),
                        "chunks": result.get('retrieved_chunks', 0)
                    }

                    # Display the main answer
                    st.markdown(result.get('answer', "Sorry, I encountered an issue processing your request."))
                    
                    # Extract sources for display
                    sources = result.get('sources', [])
                    # Show details if chunks were retrieved
                    if assistant_metrics.get('chunks', 0) > 0:
                        with st.expander("Show Details"):
                            st.markdown(f"**{assistant_metrics['emoji']} "
                                        f"Confidence: {assistant_metrics['confidence'].upper()}**")
                            st.markdown(
                                f"**Metrics:** `Search: {assistant_metrics['search_type']}` | "
                                f"`Chunks: {assistant_metrics['chunks']}` | "
                                f"`Top Match: {assistant_metrics['top_sim']:.1%}` | "
                                f"`Avg Match: {assistant_metrics['avg_sim']:.1%}`"
                            )

                            if sources:
                                st.markdown("**📚 Sources:**")
                                for s in sources:
                                    # Use .get() with default values
                                    st.markdown(
                                        f"   - **{s.get('file','?')}** (Page {s.get('page','?')}) | "
                                        f"*Similarity: {s.get('similarity',0):.1%} ({s.get('quality','?')})*"
                                    )
                            else:
                                st.markdown("**📚 No specific sources cited, but relevant document sections were found.**")
                    
                    # Append the complete assistant response to session state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result.get('answer', "Sorry, I encountered an issue processing your request."),
                        "sources": sources,
                        "metrics": assistant_metrics
                    })

                except Exception as e:
                    # Catch errors specifically during the invoke call
                    print(f"❌ Error during RAG invoke: {e}")
                    st.error(f"An error occurred while processing your request: {e}")
                    # Optionally add an error message to chat history for the user
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Sorry, I encountered an error processing your request. Please check the logs or try again. Error: {e}",
                        "sources": [],
                        "metrics": {}
                    })

# Display a warning message if the assistant failed to initialize at the start
else:
    st.warning(
        "🔴 Assistant could not be initialized. "
        "Please ensure the `build_db.py` script has been run successfully in the terminal "
        "and check the terminal logs for detailed errors."
    )
    st.info("After running `build_db.py` successfully, you may need to **refresh this page** or restart the Streamlit server.")

# Developer Footer
st.markdown(
    """
    <style>
        /* Footer styling */
        .footer {
            text-align: center;
            padding: 12px 0 6px 0;
            color: #ccc;
            font-size: 15px;
            margin-top: 25px;
        }
        .footer h4 {
            color: #00CED1;
            font-weight: 600;
            margin-bottom: 5px;
        }
        .footer a {
            color: #00CED1;
            text-decoration: none;
            font-weight: 500;
        }
        .footer a:hover {
            text-decoration: underline;
        }
        .footer small {
            display: block;
            margin-top: 4px;
            color: #888;
            font-size: 13px;
        }
    </style>

    <div class="footer">
        <h4>🤖 Contact the Developer</h4>
        <p>Connect with <strong>Ojonugwa Egwuda</strong> on 
            <a href="https://www.linkedin.com/in/egwudaojonugwa/" target="_blank">LinkedIn</a>
        </p>
        <small>© 2025 JapaPolicy AI | Built with ❤️ using Streamlit & Gemini - RAG</small>
    </div>
    """,
    unsafe_allow_html=True
)
