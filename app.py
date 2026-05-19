# 必要なライブラリのインストールコマンド:
# pip install streamlit langchain langchain-openai langchain-chroma langchain-community pypdf

import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage

# --- ページ設定 ---
st.set_page_config(page_title="RAG Chatbot", page_icon="📄")
st.title("📄 独自ドキュメント対応チャットボット (RAG)")

# --- セッションステートの初期化 ---
# 会話履歴を保持するためのリスト
if "messages" not in st.session_state:
    st.session_state.messages = []
# ベクトルストアを保持するための変数
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# --- サイドバー (ファイル管理・APIキー設定エリア) ---
with st.sidebar:
    st.header("⚙️ 設定")
    # APIキーの入力 (パスワード形式で隠す)
    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    st.divider()
    
    st.header("📂 ドキュメントのアップロード")
    # PDFファイルのアップローダー
    uploaded_file = st.file_uploader("PDFファイルをアップロードしてください", type="pdf")
    
    # インデックス作成ボタン
    if st.button("インデックス作成"):
        if not api_key:
            st.error("OpenAI API Keyを入力してください。")
        elif not uploaded_file:
            st.error("PDFファイルをアップロードしてください。")
        else:
            # 処理中のスピナーを表示
            with st.spinner("ドキュメントを処理中..."):
                try:
                    # PyPDFLoaderはファイルパスが必要なため、アップロードされたデータを一時ファイルに保存
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    # PDFの読み込み
                    loader = PyPDFLoader(tmp_file_path)
                    docs = loader.load()
                    
                    # メタデータのファイル名を元のアップロードファイル名に書き換える (参照元表示用)
                    for doc in docs:
                        doc.metadata["source"] = uploaded_file.name
                    
                    # 読み込んだドキュメントを一定サイズのチャンクに分割
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=100
                    )
                    splits = text_splitter.split_documents(docs)
                    
                    # OpenAIの埋め込みモデルを定義
                    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
                    
                    # Chromaベクトルストアを作成 (インメモリに保存)
                    vector_store = Chroma.from_documents(
                        documents=splits, 
                        embedding=embeddings
                    )
                    
                    # セッションステートに保存
                    st.session_state.vector_store = vector_store
                    
                    # 一時ファイルの削除
                    os.remove(tmp_file_path)
                    
                    st.success("インデックスの作成が完了しました！")
                except Exception as e:
                    st.error(f"エラーが発生しました: {e}")

# --- メイン画面 (チャットエリア) ---

# これまでの会話履歴を画面に表示
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# ユーザーからのチャット入力
if prompt := st.chat_input("質問を入力してください..."):
    # APIキーが未入力の場合は警告
    if not api_key:
        st.warning("サイドバーでOpenAI API Keyを入力してください。")
        st.stop()
        
    # ベクトルストアが未作成(PDF未アップロード)の場合は警告を出して終了
    if st.session_state.vector_store is None:
        st.warning("まずはサイドバーからPDFをアップロードし、インデックスを作成してください。")
        st.stop()
        
    # ユーザーの質問を画面に表示
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # ユーザーの質問を履歴に追加
    st.session_state.messages.append(HumanMessage(content=prompt))
    
    # AIの回答生成と表示
    with st.chat_message("assistant"):
        with st.spinner("回答を生成中..."):
            # LLMの定義
            llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0)
            
            # リトリーバー(検索器)の設定 (上位4件の関連ドキュメントを取得)
            retriever = st.session_state.vector_store.as_retriever(
                search_kwargs={"k": 4}
            )
            
            # システムプロンプトの設定 (指示通りに厳格な回答を求める)
            system_prompt = (
                "与えられた文脈（Context）のみに基づいて誠実に回答すること。\n"
                "文脈から判断できない場合は、知ったかぶりをせず「提示されたドキュメントからは分かりませんでした」と回答すること。\n\n"
                "{context}"
            )
            
            # QA用のプロンプトテンプレート
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            
            # ドキュメントを組み込んで回答を生成するチェーン
            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            
            # これまでの会話履歴を踏まえて、検索用の質問を再構築するプロンプト
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
            
            # 履歴を考慮したリトリーバー
            history_aware_retriever = create_history_aware_retriever(
                llm, retriever, contextualize_q_prompt
            )
            
            # 最終的なRAGチェーンの構築
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
            
            # 直近の会話履歴(今回の入力以外)を抽出してチェーンに渡す
            chat_history = st.session_state.messages[:-1]
            
            # RAGチェーンの実行
            response = rag_chain.invoke({
                "input": prompt,
                "chat_history": chat_history
            })
            
            answer = response["answer"]
            source_docs = response["context"]
            
            # 参照元(ソース)の情報を回答の末尾に追加する
            if source_docs:
                answer += "\n\n**【参照元】**\n"
                # 同じページが複数回出ないようにSetで重複排除
                sources = set()
                for doc in source_docs:
                    # メタデータからファイル名とページ番号(0始まりなので+1)を取得
                    source_name = doc.metadata.get("source", "不明なドキュメント")
                    page_num = doc.metadata.get("page", 0) + 1
                    sources.add(f"- {source_name} の {page_num} ページ")
                
                # ページ番号順などにソートして出力
                for source in sorted(list(sources)):
                    answer += f"{source}\n"
                    
            # 回答を画面に表示
            st.markdown(answer)
            
    # AIの回答を履歴に追加
    st.session_state.messages.append(AIMessage(content=answer))
