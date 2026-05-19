# 独自ドキュメント対応チャットボット (RAG)

Streamlit と LangChain を活用して構築された、PDFドキュメント対応のRAG（Retrieval-Augmented Generation）アプリケーションです。お手元のPDFをアップロードすることで、そのドキュメントの内容に基づいた誠実な回答を行うチャットボットとして機能します。

## 機能・特徴

- **PDFドキュメントの読み込み**: PDFをアップロードし、インメモリのChromaベクトルデータベースに自動でインデックス化します。
- **会話履歴の保持 (Conversational RAG)**: これまでのチャットの文脈を考慮した高度な検索と回答生成を行います。
- **情報源の明記**: 回答の末尾に、参照したPDFのファイル名とページ番号を箇条書きで提示します。
- **ハルシネーションの抑制**: ドキュメントの内容から判断できない質問には、知ったかぶりをせず「提示されたドキュメントからは分かりませんでした」と回答します。

## 必須環境

- Python 3.10以上
- OpenAI API Key

## インストール方法

以下のコマンドを実行して、必要なパッケージをインストールしてください。

```bash
pip install streamlit langchain langchain-openai langchain-chroma langchain-community langchain-classic pypdf
```

## 使い方

1. 以下のコマンドでアプリケーションを起動します。
   ```bash
   streamlit run app.py
   ```
2. ブラウザが自動的に開き、Web UIが表示されます。
3. 左側のサイドバーに **OpenAI API Key** を入力します。
4. 読み込ませたい **PDFファイル** をアップロードし、「インデックス作成」ボタンを押します。
5. メイン画面のチャット欄から、ドキュメントに関する質問を自由に入力してください！

## ライセンス (License)

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
