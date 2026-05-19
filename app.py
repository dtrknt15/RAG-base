# 必要なライブラリのインストールコマンド:
# pip install streamlit langchain langchain-openai langchain-community langchain-chroma langchain-text-splitters pypdf langchain-classic

import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
# --- 修正 ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
# --- 修正 ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# --- 修正 (langchain_classic を使用) ---
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage

# 以下のコードは変更なし
# --- ページ設定 ---
st.set_page_config(page_title="RAG Chatbot", page_icon="📄")
st.title("📄 独自ドキュメント対応チャットボット (RAG)")

# --- セッションステートの初期化 ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# --- サイドバー (ファイル管理・APIキー設定エリア) ---
with st.sidebar:
    st.header("⚙️ 設定")
    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    st.divider()
    
    st.header("📂 ドキュメントのアップロード")
    uploaded_file = st.file_uploader("PDFファイルをアップロードしてください", type="pdf")
    
    if st.button("インデックス作成"):
        if not api_key:
            st.error("OpenAI API Keyを入力してください。")
        elif not uploaded_file:
            st.error("PDFファイルをアップロードしてください。")
        else:
            with st.spinner("ドキュメントを処理中..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    loader = PyPDFLoader(tmp_file_path)
                    docs = loader.load()
                    
                    for doc in docs:
                        doc.metadata["source"] = uploaded_file.name
                    
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=100
                    )
                    splits = text_splitter.split_documents(docs)
                    
                    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
                    
                    vector_store = Chroma.from_documents(
                        documents=splits, 
                        embedding=embeddings
                    )
                    
                    st.session_state.vector_store = vector_store
                    os.remove(tmp_file_path)
                    st.success("インデックスの作成が完了しました！")
                except Exception as e:
                    st.error(f"エラーが発生しました: {e}")

# --- メイン画面 (チャットエリア) ---
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

if prompt := st.chat_input("質問を入力してください..."):
    if not api_key:
        st.warning("サイドバーでOpenAI API Keyを入力してください。")
        st.stop()
        
    if st.session_state.vector_store is None:
        st.warning("まずはサイドバーからPDFをアップロードし、インデックスを作成してください。")
        st.stop()
        
    with st.chat_message("user"):
        st.markdown(prompt)
    
    st.session_state.messages.append(HumanMessage(content=prompt))
    
    with st.chat_message("assistant"):
        with st.spinner("回答を生成中..."):
            llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0)
            
            retriever = st.session_state.vector_store.as_retriever(
                search_kwargs={"k": 4}
            )
            
            system_prompt = (
                "与えられた文脈（Context）のみに基づいて誠実に回答すること。\n"
                "文脈から判断できない場合は、知ったかぶりをせず「提示されたドキュメントからは分かりませんでした」と回答すること。\n\n"
                "{context}"
            )
            
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            
            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            
            contextualize_q_system_prompt = (
                "これまでの会話履歴と最新のユーザーの質問を踏まえて、"
                "最新の質問が以前の文脈に依存している場合は、会話履歴がなくても理解できる独立した質問に書き換えてください。"
                "質問に答える必要はなく、必要であれば書き換えるだけです。それ以外の場合はそのまま返してください。"
            )
            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            
            history_aware_retriever = create_history_aware_retriever(
                llm, retriever, contextualize_q_prompt
            )
            
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
            
            chat_history = st.session_state.messages[:-1]
            
            response = rag_chain.invoke({
                "input": prompt,
                "chat_history": chat_history
            })
            
            answer = response["answer"]
            source_docs = response["context"]
            
            if source_docs:
                answer += "\n\n**【参照元】**\n"
                sources = set()
                for doc in source_docs:
                    source_name = doc.metadata.get("source", "不明なドキュメント")
                    page_num = doc.metadata.get("page", 0) + 1
                    sources.add(f"- {source_name} の {page_num} ページ")
                
                for source in sorted(list(sources)):
                    answer += f"{source}\n"
                    
            st.markdown(answer)
            
    st.session_state.messages.append(AIMessage(content=answer))