# 必要なライブラリのインストールコマンド:
# pip install streamlit langchain langchain-openai langchain-community langchain-chroma langchain-text-splitters pypdf langchain-classic

import streamlit as st
import os
import tempfile
import dotenv
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

# 環境変数の読み込み
dotenv.load_dotenv()

# 以下のコードは変更なし
# --- ページ設定 ---
st.set_page_config(page_title="RAG Chatbot", page_icon="📄")

# --- UIカスタムCSSの注入 ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;800&family=Inter:wght@400;500;600&display=swap');

/* 全体のフォント設定 */
.stApp {
    font-family: 'Inter', sans-serif;
}

/* タイトルのグラデーションとフォント */
h1 {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #6366f1, #06b6d4, #3b82f6) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    padding-bottom: 10px !important;
}

/* チャットメッセージの見た目洗練 */
div[data-testid="stChatMessage"] {
    animation: slideUp 0.4s ease-out;
    border-radius: 16px !important;
    border: 1px solid rgba(0, 0, 0, 0.05) !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.02) !important;
    margin-bottom: 16px !important;
}

/* サイドバーの背景トーン */
section[data-testid="stSidebar"] {
    background-color: #f8fafc !important;
    border-right: 1px solid #f1f5f9 !important;
}

/* 参照元のアコーディオンをスタイリッシュに */
div[data-testid="stExpander"] {
    border: 1px solid rgba(0, 0, 0, 0.06) !important;
    border-radius: 10px !important;
    background-color: rgba(255, 255, 255, 0.4) !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.01) !important;
}

/* アニメーションキーフレーム */
@keyframes slideUp {
    from {
        opacity: 0;
        transform: translateY(12px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Streamlitのデフォルトボタンホバー調整 */
button[kind="secondary"] {
    transition: all 0.25s ease !important;
    border-radius: 8px !important;
}
button[kind="secondary"]:hover {
    border-color: #6366f1 !important;
    color: #6366f1 !important;
    transform: translateY(-1px);
}
</style>
""", unsafe_allow_html=True)

# タイトルと会話クリアボタンのレイアウト
col1, col2 = st.columns([0.75, 0.25])
with col1:
    st.title("📄 独自ドキュメント対応チャットボット (RAG)")
with col2:
    st.write("")  # 余白調整
    st.write("")  # 余白調整
    if st.button("🗑️ 会話クリア", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- 定数とヘルパー関数 ---
PERSIST_DIR = "./chroma_db"

@st.cache_data(show_spinner="利用可能なモデルを取得中...")
def fetch_models(api_key, base_url):
    if not api_key:
        return ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=5.0)
        models_list = client.models.list()
        ids = [m.id for m in models_list.data]
        return sorted(ids)
    except Exception as e:
        st.sidebar.warning(f"モデル一覧の取得に失敗しました (OpenAIのデフォルトモデルを表示します): {e}")
        return ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]

def get_indexed_files():
    if st.session_state.vector_store is None:
        return []
    try:
        # Chromaのget()メソッドで全メタデータのソース一覧を取得
        data = st.session_state.vector_store.get(include=["metadatas"])
        metadatas = data.get("metadatas", [])
        sources = set()
        for meta in metadatas:
            if meta and "source" in meta:
                sources.add(meta["source"])
        return sorted(list(sources))
    except Exception as e:
        return []

def get_vector_store(api_key, base_url, embedding_model):
    # すでに読み込み済みで設定が変わっていなければそのまま返す
    if (st.session_state.vector_store is not None and
        st.session_state.get("last_api_key") == api_key and
        st.session_state.get("last_base_url") == base_url and
        st.session_state.get("last_embedding_model") == embedding_model):
        return st.session_state.vector_store

    # 新しく読み込むか再初期化する
    if api_key and os.path.exists(PERSIST_DIR) and len(os.listdir(PERSIST_DIR)) > 0:
        try:
            embeddings = OpenAIEmbeddings(model=embedding_model, api_key=api_key, base_url=base_url)
            vector_store = Chroma(
                persist_directory=PERSIST_DIR,
                embedding_function=embeddings
            )
            st.session_state.vector_store = vector_store
            st.session_state.last_api_key = api_key
            st.session_state.last_base_url = base_url
            st.session_state.last_embedding_model = embedding_model
            return vector_store
        except Exception as e:
            st.sidebar.error(f"永続化インデックスの読込エラー: {e}")
            return None
    return st.session_state.vector_store

# --- セッションステートの初期化 ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# --- サイドバー (ファイル管理・APIキー設定エリア) ---
with st.sidebar:
    st.header("⚙️ 設定")
    default_api_key = os.environ.get("OPENAI_API_KEY", "")
    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...", value=default_api_key)
    
    default_base_url = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    base_url = st.text_input("Base URL", placeholder="https://api.openai.com/v1", value=default_base_url)
    
    embedding_model = st.text_input("埋め込みモデル名", value="text-embedding-3-small")
    
    use_custom_model = st.checkbox("カスタムモデル名を手動入力", value=False)
    if use_custom_model:
        selected_model = st.text_input("モデル名", value="gpt-4o-mini")
    else:
        models = fetch_models(api_key, base_url)
        default_model = "gpt-4o-mini"
        if default_model not in models and models:
            default_model = models[0]
        selected_model = st.selectbox(
            "利用するモデル", 
            options=models, 
            index=models.index(default_model) if default_model in models else 0
        )
    
    # ベクトルデータベースの自動読み込み/同期
    if api_key:
        get_vector_store(api_key, base_url, embedding_model)
        
    prompt_mode = st.radio(
        "回答モード",
        options=["厳格に文書回答", "一般知識も交える"],
        index=0,
        help="厳格モード：ドキュメント内の情報のみに基づいて回答します。\n一般知識併用モード：ドキュメントにない情報も一般知識から補完して回答します。"
    )
        
    with st.expander("⚙️ 詳細RAG設定"):
        chunk_size = st.slider("チャンクサイズ", min_value=100, max_value=3000, value=1000, step=100)
        chunk_overlap = st.slider("重複サイズ (overlap)", min_value=0, max_value=500, value=100, step=10)
        search_type = st.selectbox("検索タイプ", options=["similarity", "mmr"], index=0)
        k_value = st.slider("取得ドキュメント数 (k)", min_value=1, max_value=10, value=4, step=1)
                
    st.divider()
    
    st.header("📂 ドキュメントのアップロード")
    uploaded_files = st.file_uploader("PDFファイルをアップロードしてください", type="pdf", accept_multiple_files=True)
    
    if st.button("インデックス作成"):
        if not api_key:
            st.error("OpenAI API Keyを入力してください。")
        elif not uploaded_files:
            st.error("PDFファイルをアップロードしてください。")
        else:
            with st.spinner("ドキュメントを処理中..."):
                try:
                    embeddings = OpenAIEmbeddings(model=embedding_model, api_key=api_key, base_url=base_url)
                    
                    st.session_state.vector_store = Chroma(
                        persist_directory=PERSIST_DIR,
                        embedding_function=embeddings
                    )
                    st.session_state.last_api_key = api_key
                    st.session_state.last_base_url = base_url
                    st.session_state.last_embedding_model = embedding_model
                    
                    indexed_files = get_indexed_files()
                    
                    for uploaded_file in uploaded_files:
                        # すでに同じ名前のファイルがインデックスされている場合、古い情報を削除
                        if uploaded_file.name in indexed_files:
                            try:
                                data = st.session_state.vector_store.get(where={"source": uploaded_file.name})
                                ids = data.get("ids", [])
                                if ids:
                                    st.session_state.vector_store.delete(ids=ids)
                            except Exception as e:
                                pass
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name
                        
                        try:
                            loader = PyPDFLoader(tmp_file_path)
                            docs = loader.load()
                            
                            for doc in docs:
                                doc.metadata["source"] = uploaded_file.name
                            
                            text_splitter = RecursiveCharacterTextSplitter(
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap
                            )
                            splits = text_splitter.split_documents(docs)
                            st.session_state.vector_store.add_documents(documents=splits)
                        finally:
                            if os.path.exists(tmp_file_path):
                                os.remove(tmp_file_path)
                                
                    st.success("インデックスの作成・更新が完了しました！")
                    st.rerun()
                except Exception as e:
                    st.error(f"エラーが発生しました: {e}")
                    
    # --- インデックス済みファイル一覧 ---
    st.divider()
    st.header("🗂️ インデックス済みファイル")
    indexed_files = get_indexed_files()
    
    # 統計情報の算出
    total_pages = 0
    total_chunks = 0
    if st.session_state.vector_store is not None:
        try:
            data = st.session_state.vector_store.get(include=["metadatas"])
            metadatas = data.get("metadatas", [])
            total_chunks = len(metadatas)
            
            pages_per_source = {}
            for meta in metadatas:
                if meta and "source" in meta and "page" in meta:
                    source_name = meta["source"]
                    p_num = meta["page"]
                    if source_name not in pages_per_source:
                        pages_per_source[source_name] = set()
                    pages_per_source[source_name].add(p_num)
            total_pages = sum(len(pages) for pages in pages_per_source.values())
        except Exception as e:
            pass

    # 統計カードの表示
    if indexed_files:
        st.markdown(f"""
        <div style="display: flex; gap: 10px; margin-bottom: 15px;">
            <div style="flex: 1; background: white; padding: 12px; border-radius: 10px; border: 1px solid rgba(0,0,0,0.06); text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.01);">
                <div style="font-size: 0.8rem; color: #64748b; font-weight: 500;">総ページ数</div>
                <div style="font-size: 1.4rem; color: #6366f1; font-weight: 800; margin-top: 4px;">{total_pages}</div>
            </div>
            <div style="flex: 1; background: white; padding: 12px; border-radius: 10px; border: 1px solid rgba(0,0,0,0.06); text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.01);">
                <div style="font-size: 0.8rem; color: #64748b; font-weight: 500;">総チャンク数</div>
                <div style="font-size: 1.4rem; color: #06b6d4; font-weight: 800; margin-top: 4px;">{total_chunks}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        for f_name in indexed_files:
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                st.write(f"📄 {f_name}")
            with col2:
                if st.button("🗑️", key=f"del_{f_name}"):
                    try:
                        data = st.session_state.vector_store.get(where={"source": f_name})
                        ids = data.get("ids", [])
                        if ids:
                            st.session_state.vector_store.delete(ids=ids)
                            st.success(f"{f_name} を削除しました。")
                            st.rerun()
                        else:
                            st.warning(f"{f_name} のデータが見つかりませんでした。")
                    except Exception as e:
                        st.error(f"削除エラー: {e}")
    else:
        st.info("インデックス済みのファイルはありません。")

# --- メイン画面 (チャットエリア) ---
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)
            # 参照元の表示 (st.expander)
            sources = msg.additional_kwargs.get("sources", [])
            if sources:
                with st.expander("🔍 参照元を確認する"):
                    for src in sources:
                        st.markdown(src)

if prompt := st.chat_input("質問を入力してください..."):
    if not api_key:
        st.warning("サイドバーでOpenAI API Keyを入力してください。")
        st.stop()
        
    if st.session_state.vector_store is None or not get_indexed_files():
        st.warning("まずはサイドバーからPDFをアップロードし、インデックスを作成してください。")
        st.stop()
        
    with st.chat_message("user"):
        st.markdown(prompt)
    
    st.session_state.messages.append(HumanMessage(content=prompt))
    
    with st.chat_message("assistant"):
        with st.spinner("回答を生成中..."):
            llm = ChatOpenAI(model=selected_model, api_key=api_key, base_url=base_url, temperature=0)
            
            retriever = st.session_state.vector_store.as_retriever(
                search_type=search_type,
                search_kwargs={"k": k_value}
            )
            
            if prompt_mode == "厳格に文書回答":
                system_prompt = (
                    "与えられた文脈（Context）のみに基づいて誠実に回答すること。\n"
                    "文脈から判断できない場合は、知ったかぶりをせず「提示されたドキュメントからは分かりませんでした」と回答すること。\n\n"
                    "{context}"
                )
            else:
                system_prompt = (
                    "与えられた文脈（Context）を最も重要な情報源として回答してください。\n"
                    "もし文脈だけでは質問に十分に答えられない場合は、あなたの持つ一般的な知識も交えて分かりやすく回答してください。\n"
                    "ただし、文脈に含まれていない一般的な知識に基づく情報を提供する場合は、回答の中で「※提示されたドキュメント以外の知識に基づきます」などの補足を明記してください。\n\n"
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
            
            # 参照元の抽出
            sources_list = []
            if source_docs:
                sources = set()
                for doc in source_docs:
                    source_name = doc.metadata.get("source", "不明なドキュメント")
                    page_num = doc.metadata.get("page", 0) + 1
                    sources.add(f"- {source_name} の {page_num} ページ")
                sources_list = sorted(list(sources))
            
            st.markdown(answer)
            if sources_list:
                with st.expander("🔍 参照元を確認する"):
                    for src in sources_list:
                        st.markdown(src)
            
    st.session_state.messages.append(AIMessage(content=answer, additional_kwargs={"sources": sources_list}))