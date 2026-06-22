import os
import streamlit as st
import fitz  # PyMuPDF
from typing import List
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.schema import Document
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator
from llama_index.core.node_parser import SentenceSplitter

# 1. 환경 변수 로드 (.env 파일 읽기)
load_dotenv(override=True) 

# 2. 페이지 설정 및 UI 초기화
st.set_page_config(page_title="소설 AI", page_icon="📖", layout="wide")
st.title("📖 소설 Q&A 봇")
st.caption("PDF 소설을 업로드하고, 읽은 권/페이지/줄 까지만 질문하세요.")

if "index" not in st.session_state:
    st.session_state.index = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "volume_to_max_pages" not in st.session_state:
    st.session_state.volume_to_max_pages = {}
if "page_to_max_lines" not in st.session_state:
    st.session_state.page_to_max_lines = {}

# --- 로컬 임베딩 모델 캐싱 ---
@st.cache_resource(show_spinner=False)
def load_local_embedding_model():
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    return HuggingFaceEmbedding(model_name="BAAI/bge-m3")

# 3. 사이드바: 설정 및 파일 업로드
with st.sidebar:
    st.header("⚙️ 소설 설정 ")
    
    uploaded_files = st.file_uploader("소설 PDF 다중 업로드", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        uploaded_files = sorted(uploaded_files, key=lambda x: x.name)
        current_file_names = [f.name for f in uploaded_files]
        
        if "last_file_names" not in st.session_state or current_file_names != st.session_state.last_file_names:
            st.session_state.index = None
            st.session_state.last_file_names = current_file_names
            st.session_state.volume_to_max_pages = {}
            st.session_state.page_to_max_lines = {}
            st.rerun() 
            
    # 권 별 최대 페이지 수 계산 
    if uploaded_files and not st.session_state.volume_to_max_pages:
        with st.spinner("문서 구조를 빠르게 스캔 중입니다..."):
            for idx, uploaded_file in enumerate(uploaded_files):
                vol_num = idx + 1
                doc_pdf = fitz.open(stream=uploaded_file.getvalue(), filetype="pdf")
                st.session_state.volume_to_max_pages[vol_num] = len(doc_pdf)
                st.session_state.page_to_max_lines[vol_num] = {}
                
                for page_idx, page in enumerate(doc_pdf):
                    page_num = page_idx + 1
                    text_instances = page.get_text("blocks")
                    
                    line_count = 0
                    for block in text_instances:
                        block_text = block[4].strip()
                        if not block_text:
                            continue
                        
                        lines = block_text.split('\n')
                        for line_text in lines:
                            if len(line_text.strip()) >= 2:
                                line_count += 1
                                
                    # 빈 페이지나 그림만 있는 페이지의 경우 UI 오류(min > max) 방지를 위해 최소 1로 설정
                    st.session_state.page_to_max_lines[vol_num][page_num] = max(1, line_count)
                    
                doc_pdf.close()
    
    if uploaded_files and st.session_state.index is None:
        Settings.llm = GoogleGenAI(model="gemini-2.5-flash", temperature=0.1)
        
        with st.spinner("로컬 임베딩 모델을 로드 중입니다... (최초 1회 다운로드)"):
            Settings.embed_model = load_local_embedding_model()
            node_parser = SentenceSplitter(chunk_size=350, chunk_overlap=30)

        PERSIST_DIR = "./storage_local_context_line"
        
        # 이미 저장된 인덱스가 존재하는지 확인
        if os.path.exists(PERSIST_DIR):
            with st.spinner("저장된 인덱스를 불러오는 중입니다... (1초 소요)"):
                storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
                st.session_state.index = load_index_from_storage(storage_context)
                st.success("저장된 로컬 인덱스 로드 완료!")
        else:
            with st.spinner("PDF 파일을 줄(Line) 단위로 분석하는 중입니다... (최초 1회)"):
                all_raw_documents = []
                
                # 파일별로 권, 페이지, 줄 번호를 추출하여 메타데이터에 입력
                for idx, uploaded_file in enumerate(uploaded_files):
                    vol_num = idx + 1
                    doc_pdf = fitz.open(stream=uploaded_file.getvalue(), filetype="pdf")
                    
                    for page_idx, page in enumerate(doc_pdf):
                        page_num = page_idx + 1
                        text_instances = page.get_text("blocks")
                        
                        line_counter = 1
                        for block in text_instances:
                            block_text = block[4].strip()
                            if not block_text:
                                continue
                            
                            lines = block_text.split('\n')
                            for line_text in lines:
                                line_text = line_text.strip()
                                if len(line_text) < 2: # 의미 없는 짧은 기호/공백 제외
                                    continue
                                
                                doc = Document(
                                    text=line_text,
                                    metadata={
                                        "volume": vol_num,
                                        "page": page_num,
                                        "line": line_counter
                                    }
                                )
                                all_raw_documents.append(doc)
                                line_counter += 1
                    doc_pdf.close()

                nodes = node_parser.get_nodes_from_documents(all_raw_documents)

                for node in nodes:
                    v = node.metadata.get("volume", 1)
                    p = node.metadata.get("page", 1)
                    l = node.metadata.get("line", 1) # 이 단락이 시작되는 줄 번호
                    
                    # 단락의 시작 줄 번호를 기반으로 고유 검색 주소 계산
                    node.metadata["read_index"] = (v * 100000000) + (p * 10000) + l
            
            # --- 진행률 표시와 함께 순차적 임베딩 시작 ---
            st.session_state.index = VectorStoreIndex([]) 
            
            progress_text = "맥락 인덱싱 중..."
            progress_bar = st.progress(0.0, text=progress_text)
            
            batch_size = 100  # 짧은 문장들이므로 배치 사이즈를 늘려 처리 속도 향상
            total_nodes = len(nodes)
            
            for i in range(0, total_nodes, batch_size):
                batch = nodes[i : i + batch_size]
                st.session_state.index.insert_nodes(batch)
                progress = min((i + batch_size) / total_nodes, 1.0)
                progress_bar.progress(progress, text=f"{progress_text} ({min(i + batch_size, total_nodes)} / {total_nodes} 완료)")
            
            # 완료된 인덱스를 디스크에 영구 저장
            st.session_state.index.storage_context.persist(persist_dir=PERSIST_DIR)
            st.success(f"로컬 정밀 인덱싱 및 디스크 저장 성공! (총 {total_nodes}문장)")

    # 사이드바: 읽은 위치 (권, 페이지, 줄) 입력창
    current_volume = 1
    current_page = 1
    current_line = 1
    
    if uploaded_files and st.session_state.volume_to_max_pages:
        st.divider()
        st.subheader("📖 현재 읽은 위치")
        
        vol_options = list(st.session_state.volume_to_max_pages.keys())
        current_volume = st.selectbox(
            "권 선택", 
            options=vol_options,
            format_func=lambda x: f"{x}권 ({uploaded_files[x-1].name})"
        )
        
        max_pages_in_vol = st.session_state.volume_to_max_pages[current_volume]
        current_page = st.number_input(
            f"선택한 권의 읽은 페이지 (최대 {max_pages_in_vol}p)", 
            min_value=1, 
            max_value=max_pages_in_vol,
            value=1,
            step=1
        )
        
        max_lines_in_page = st.session_state.page_to_max_lines[current_volume][current_page]
        
        current_line = st.number_input(
            f"현재 페이지의 읽은 줄 번호 (최대 {max_lines_in_page}줄)",
            min_value=1,
            max_value=max_lines_in_page,
            value=min(30, max_lines_in_page), # 페이지에 줄이 적을 경우를 대비해 초기값 안전장치
            step=1,
            help=f"이 페이지는 총 {max_lines_in_page}줄로 이루어져 있습니다."
        )

# 4. 메인 화면: 채팅 인터페이스
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("소설 내용에 대해 질문해 보세요!"):
    # 예외 처리: 파일 미업로드 시 채팅 제한
    if st.session_state.index is None:
        st.warning("먼저 왼쪽 사이드바에서 PDF 파일을 업로드해 주세요.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("기억을 더듬는 중..."):
                
                current_read_index = (current_volume * 100000000) + (current_page * 10000) + current_line

                filters = MetadataFilters(
                    filters=[
                        MetadataFilter(
                            key="read_index", 
                            value=current_read_index, 
                            operator=FilterOperator.LTE  # 작거나 같다(<=)
                        )
                    ]
                )

                query_engine = st.session_state.index.as_query_engine(
                    filters=filters,
                    similarity_top_k=5 # 이미 스포일러가 제거된 안전한 문서 중에서만 검색하므로 기본값 유지 가능
                )
                
                try:
                    response = query_engine.query(prompt)
                    
                    if not str(response).strip() or str(response) == "Empty Response":
                        bot_reply = "현재까지 읽으신 분량 내에서는 해당 질문에 대한 내용을 찾을 수 없습니다."
                        st.markdown(bot_reply)
                        st.session_state.messages.append({"role": "assistant", "content": bot_reply})
                    else:
                        st.markdown(response.response)
                        st.session_state.messages.append({"role": "assistant", "content": response.response})
                        
                except Exception as e:
                    error_msg = str(e)
                    if "503" in error_msg or "UNAVAILABLE" in error_msg:
                        st.error("🚨 구글 AI 서버에 일시적인 트래픽 과부하가 발생했습니다. 잠시 후 다시 시도해 주세요.")
                    elif "429" in error_msg or "Quota" in error_msg:
                        st.error("🚨 구글 API 무료 할당량을 초과했습니다. 잠시 후 다시 시도해 주세요.")
                    else:
                        st.error(f"예기치 못한 오류가 발생했습니다: {error_msg}")