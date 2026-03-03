import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """You are Research-Buddy, a warm and encouraging AI academic assistant \
specialising in peer-reviewed literature, official datasets and open-government data.

Your role is to guide students through every step of the research process — clarifying \
their question, finding and evaluating sources, synthesising literature, designing methods \
and structuring outputs — in a natural, flowing back-and-forth conversation.

How you behave:
- Be concise, scholarly but accessible. Use short paragraphs, bullets or numbered steps \
where they genuinely help; avoid walls of text.
- Always respond to what the student said, then naturally suggest the next step, a \
follow-up question, or an alternative angle. Never give a response that just ends — \
always give the student something to react to or build on.
- When a topic is first raised, ask one or two focused questions to understand the \
student's level (undergraduate, master's, PhD, etc.) and what they need to produce \
(essay, literature review, data analysis, poster, etc.).
- When citing sources, provide: author(s), year, title, journal/publisher, DOI or URL, \
and a one-sentence relevance note.
- In every response, already make something useful for the student to work with. For example: start drafting an outline, suggest search keywords, sketch a data analysis plan, or draft a paragraph. \
- Distinguish clearly between direct quotations, paraphrases and your own synthesis.
- If you are uncertain or no high-quality source exists, say so — never fabricate.
- Do not write entire graded assignments; coach, explain and exemplify instead.
- If uploaded documents are available in the context, ground your answers in them first, \
then supplement with your broader knowledge.

Always close each response with a natural conversational prompt — a clarifying question, \
a suggested next step, or a short menu of options (e.g., "Shall I draft some search \
keywords, or would you prefer to start with an outline?"). Keep the dialogue moving.

Relevant document excerpts:
{context}"""

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])

load_dotenv()

st.set_page_config(page_title="Research Assistant", page_icon="🔬")
st.title("Research Assistant")

# --- Sidebar: API key input ---
with st.sidebar:
    st.header("Configuration")
    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Your key is only stored in this browser session and never sent anywhere except OpenAI.",
    )

# Key resolution: sidebar > Streamlit secrets > .env
api_key = (
    api_key_input
    or st.secrets.get("OPENAI_API_KEY", "")
    or os.getenv("OPENAI_API_KEY", "")
)

if not api_key:
    st.info("Enter your OpenAI API key in the sidebar to get started.")
    st.stop()

# --- Tabs ---
tab_docs, tab_chat = st.tabs(["Documents", "Chat"])


def build_chain(uploaded_files, key):
    """Load, chunk, and embed uploaded PDFs into an in-memory vectorstore, then return a RAG chain."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=key)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        is_separator_regex=True,
        separators=[r"\n\n+", r"\n+", r"\s+"],
    )

    all_splits = []
    progress = st.progress(0, text="Loading documents...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, uploaded_file in enumerate(uploaded_files):
            progress.progress(
                int((i / len(uploaded_files)) * 50),
                text=f"Loading {uploaded_file.name}...",
            )
            tmp_path = os.path.join(tmp_dir, uploaded_file.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.read())
            docs = PyPDFLoader(tmp_path).load()
            all_splits.extend(text_splitter.split_documents(docs))

    if not all_splits:
        progress.empty()
        return None

    progress.progress(50, text=f"Embedding {len(all_splits)} chunks...")

    # In-memory vectorstore — no persist_directory, nothing written to disk
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)

    progress.progress(90, text="Building retrieval chain...")

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.5},
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="o3", api_key=key),
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
    )

    progress.progress(100, text="Done!")
    progress.empty()
    return chain


# --- Documents Tab ---
with tab_docs:
    st.header("Upload Documents")
    st.write("Upload one or more PDF files to build the knowledge base.")

    uploaded_files = st.file_uploader(
        "Select PDF files",
        type="pdf",
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.write(f"**{len(uploaded_files)} file(s) selected:**")
        for f in uploaded_files:
            st.markdown(f"- {f.name}")

        if st.button("Index Documents", type="primary"):
            chain = build_chain(uploaded_files, api_key)
            if chain:
                st.session_state.chain = chain
                st.session_state.messages = [
                    {
                        "role": "assistant",
                        "content": (
                            "Hi! I'm **Research-Buddy**, your academic assistant. "
                            "Your documents are loaded and I'm ready to help.\n\n"
                            "To get started — what are you working on, and what do you "
                            "need to produce? (e.g., a literature review, an essay outline, "
                            "a methodology section, or something else?)"
                        ),
                    }
                ]
                st.success(
                    f"Indexed {len(uploaded_files)} file(s). "
                    "Switch to the **Chat** tab to start asking questions."
                )
            else:
                st.error("No content could be extracted from the uploaded files.")

    if "chain" in st.session_state:
        st.info(
            "A knowledge base is already loaded. "
            "Upload new files and click **Index Documents** to replace it."
        )


# --- Chat Tab ---
with tab_chat:
    if "chain" not in st.session_state:
        st.info("Upload and index documents in the **Documents** tab first.")
    else:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Scrollable message history — keeps the input pinned below it
        message_area = st.container(height=550, border=False)
        with message_area:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        if prompt := st.chat_input("Ask a question about your documents..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with message_area:
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Searching..."):
                        result = st.session_state.chain.invoke({"question": prompt})
                        answer = result["answer"]
                        sources = result.get("source_documents", [])

                    st.markdown(answer)

                    if sources:
                        with st.expander(f"Sources ({len(sources)})"):
                            seen = set()
                            for doc in sources:
                                src = doc.metadata.get("source", "Unknown")
                                page = doc.metadata.get("page", "")
                                label = os.path.basename(src) + (
                                    f" — page {page + 1}" if page != "" else ""
                                )
                                if label not in seen:
                                    seen.add(label)
                                    st.markdown(f"- {label}")

            st.session_state.messages.append({"role": "assistant", "content": answer})
