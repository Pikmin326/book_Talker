import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator

# 1. 환경 변수 로드 (.env 파일 읽기)
load_dotenv() # .env 파일에 있는 GOOGLE_API_KEY를 자동으로 시스템 환경변수로 불러옵니다.

# 2. 페이지 설정 및 UI 초기화
st.set_page_config(page_title="소설 AI", page_icon="📖", layout="wide")
st.title("📖 소설 Q&A 봇")
st.caption("PDF 소설을 업로드하고, 읽은 페이지까지만 질문하세요.")

if "index" not in st.session_state:
    st.session_state.index = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_pages" not in st.session_state:
    st.session_state.total_pages = 0

# 3. 사이드바: 설정 및 파일 업로드
with st.sidebar:
    st.header("⚙️ 소설 설정 ")
    # 기존에 있던 API 키 입력 UI 제거됨
    
    uploaded_file = st.file_uploader("소설 PDF 업로드", type=["pdf"])
    
    # 1. 파일이 변경되었는지 확인하는 로직 추가
    if "last_file_name" not in st.session_state:
        st.session_state.last_file_name = None

    # 업로드된 파일이 있고, 파일이 바뀌었다면(또는 처음 업로드라면)
    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.last_file_name:
            # 상태 초기화
            st.session_state.index = None
            st.session_state.last_file_name = uploaded_file.name
            st.rerun() # 앱을 새로고침하여 로직 재실행
    
    # 파일이 업로드되었고, 아직 인덱싱 전일 때 실행
    if uploaded_file and st.session_state.index is None:
        # API 키는 .env에서 자동으로 로드되었으므로 바로 모델 설정 진행
        Settings.llm = GoogleGenAI(model="gemini-2.5-flash", temperature=0.1)
        Settings.embed_model = GoogleGenAIEmbedding(model_name="gemini-embedding-2")

        with st.spinner("문서를 읽고 인덱싱하는 중입니다... 잠시만 기다려주세요."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            loader = PyMuPDFReader()
            documents = loader.load(file_path=tmp_file_path)
            
            st.session_state.total_pages = len(documents)
            
            for i, doc in enumerate(documents):
                doc.metadata["page"] = i + 1 
            
            st.session_state.index = VectorStoreIndex.from_documents(documents)
            st.success(f"인덱싱 완료! (총 {st.session_state.total_pages}페이지)")
            os.remove(tmp_file_path)
            

    st.divider()
    
    # 슬라이더 대신 숫자 타이핑(number_input)으로 변경
    current_page = st.number_input(
        "현재 읽은 페이지 입력", 
        min_value=1, 
        value=1,
        step=1,
        help="키보드로 숫자를 입력하세요. 이 페이지 이하의 내용만 참고하여 답변합니다."
    )

    # 입력된 페이지가 문서의 총 페이지를 초과하는지 검사
    page_error = False
    if st.session_state.index is not None:
        if current_page > st.session_state.total_pages:
            st.error(f"오류: 입력한 페이지({current_page}p)가 문서의 총 페이지({st.session_state.total_pages}p)보다 큽니다. 올바른 숫자를 입력해주세요.")
            page_error = True

# 4. 메인 화면: 채팅 인터페이스
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("소설 내용에 대해 질문해 보세요!"):
    # 예외 처리: 파일 미업로드 또는 페이지 설정 오류 시 채팅 제한
    if st.session_state.index is None:
        st.warning("먼저 왼쪽 사이드바에서 PDF 파일을 업로드해 주세요.")
    elif page_error:
        st.warning("현재 읽은 페이지 설정에 오류가 있습니다. 올바른 숫자를 입력한 뒤 다시 질문해 주세요.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("생각 중..."):
                filters = MetadataFilters(
                    filters=[
                        MetadataFilter(
                            key="page", 
                            value=current_page, 
                            operator=FilterOperator.LTE
                        )
                    ]
                )
                
                query_engine = st.session_state.index.as_query_engine(filters=filters)
                response = query_engine.query(prompt)
                
                st.markdown(response.response)
                
        st.session_state.messages.append({"role": "assistant", "content": response.response})