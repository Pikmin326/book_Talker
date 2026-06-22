import os
import json
import streamlit as st
import fitz  # PyMuPDF
import shutil
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.schema import Document
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator

# 1. 환경 변수 로드 (.env 파일 읽기)
load_dotenv(override=True) 

# 2. 페이지 설정 및 UI 초기화
st.set_page_config(page_title="소설 AI", page_icon="📖", layout="wide")
st.title("📖 소설 Q&A 봇")
st.caption("초정밀 RAG: 전후 3페이지의 문맥을 파악하되, 스포일러는 원천 차단합니다.")

if "index" not in st.session_state:
    st.session_state.index = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "all_lines" not in st.session_state:
    st.session_state.all_lines = []

# --- 로컬 임베딩 모델 캐싱 ---
@st.cache_resource(show_spinner=False)
def load_local_embedding_model():
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    return HuggingFaceEmbedding(model_name="BAAI/bge-m3")

# 3. 사이드바: 설정 및 파일 업로드
with st.sidebar:
    st.header("⚙️ 소설 설정 ")
    
    uploaded_files = st.file_uploader("소설 PDF 다중 업로드", type=["pdf"], accept_multiple_files=True)

    # 새로운 컨텍스트 방식 전용 저장소
    PERSIST_DIR = "./storage_novel_strict_rag"
    LINES_DB_PATH = os.path.join(PERSIST_DIR, "all_lines_db.json")

    if uploaded_files:
        uploaded_files = sorted(uploaded_files, key=lambda x: x.name)
        current_file_names = [f.name for f in uploaded_files]
        
        if "last_file_names" not in st.session_state or current_file_names != st.session_state.last_file_names:
            st.session_state.index = None
            st.session_state.last_file_names = current_file_names
            st.session_state.all_lines = []
            
            if os.path.exists(PERSIST_DIR):
                shutil.rmtree(PERSIST_DIR)
                
            st.rerun()
    
    if uploaded_files and st.session_state.index is None:
        Settings.llm = GoogleGenAI(model="gemini-2.5-flash", temperature=0.1)
        
        with st.spinner("로컬 임베딩 모델을 로드 중입니다..."):
            Settings.embed_model = load_local_embedding_model()
        
        if os.path.exists(PERSIST_DIR) and os.path.exists(LINES_DB_PATH):
            with st.spinner("저장된 인덱스와 문맥 DB를 불러오는 중입니다..."):
                storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
                st.session_state.index = load_index_from_storage(storage_context)
                
                with open(LINES_DB_PATH, "r", encoding="utf-8") as f:
                    st.session_state.all_lines = json.load(f)
                st.success("데이터 로드 완료!")
        else:
            with st.spinner("소설을 한 줄 단위로 정밀 분석 중입니다... (최초 1회)"):
                all_lines = []
                
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
                                if len(line_text) < 2:
                                    continue
                                
                                read_index_val = (vol_num * 100000000) + (page_num * 10000) + line_counter
                                
                                # 모든 줄을 원본 DB 배열에 저장
                                all_lines.append({
                                    "read_index": read_index_val,
                                    "vol": vol_num,
                                    "page": page_num,
                                    "line": line_counter,
                                    "text": line_text
                                })
                                line_counter += 1
                    doc_pdf.close()
                
                st.session_state.all_lines = all_lines
                
                # 저장소 폴더 생성 후 원본 문맥 DB 저장
                os.makedirs(PERSIST_DIR, exist_ok=True)
                with open(LINES_DB_PATH, "w", encoding="utf-8") as f:
                    json.dump(all_lines, f, ensure_ascii=False)
                
                # 벡터 인덱싱을 위한 소형 청크 생성 (약 10줄 단위)
                nodes = []
                chunk_size = 10 
                for i in range(0, len(all_lines), chunk_size):
                    chunk_data = all_lines[i : i + chunk_size]
                    chunk_text = " ".join([x["text"] for x in chunk_data])
                    
                    doc = Document(
                        text=chunk_text,
                        metadata={
                            "array_start_idx": i, # DB 내의 시작 위치
                            "read_index": chunk_data[0]["read_index"] # 필터링 기준 위치
                        }
                    )
                    nodes.append(doc)
            
            st.session_state.index = VectorStoreIndex([]) 
            
            progress_text = "정밀 인덱싱 진행 중..."
            progress_bar = st.progress(0.0, text=progress_text)
            
            batch_size = 100  
            total_nodes = len(nodes)
            
            for i in range(0, total_nodes, batch_size):
                batch = nodes[i : i + batch_size]
                st.session_state.index.insert_nodes(batch)
                progress = min((i + batch_size) / total_nodes, 1.0)
                progress_bar.progress(progress, text=f"{progress_text} ({min(i + batch_size, total_nodes)} / {total_nodes} 완료)")
            
            st.session_state.index.storage_context.persist(persist_dir=PERSIST_DIR)
            st.success("인덱싱 및 디스크 저장 완료!")

    # UI 구성을 위해 all_lines에서 권/페이지 정보 동적 추출 (매번 PDF를 읽지 않기 위함)
    volume_to_max_pages = {}
    page_to_max_lines = {}
    
    if st.session_state.all_lines:
        for item in st.session_state.all_lines:
            v, p, l = item["vol"], item["page"], item["line"]
            volume_to_max_pages[v] = max(volume_to_max_pages.get(v, 0), p)
            if v not in page_to_max_lines:
                page_to_max_lines[v] = {}
            page_to_max_lines[v][p] = max(page_to_max_lines[v].get(p, 0), l)

    # 사이드바 입력 컴포넌트
    current_volume = 1
    current_page = 1
    current_line = 1 
    
    if uploaded_files and volume_to_max_pages:
        st.divider()
        st.subheader("📖 현재 읽은 위치")
        
        vol_options = list(volume_to_max_pages.keys())
        current_volume = st.selectbox(
            "권 선택", 
            options=vol_options,
            format_func=lambda x: f"{x}권 ({uploaded_files[x-1].name})"
        )
        
        max_pages_in_vol = volume_to_max_pages[current_volume]
        current_page = st.number_input(
            f"선택한 권의 읽은 페이지 (최대 {max_pages_in_vol}p)", 
            min_value=1, 
            max_value=max_pages_in_vol,
            value=1,
            step=1
        )
        
        max_lines_in_page = page_to_max_lines[current_volume][current_page]
        current_line = st.number_input(
            f"현재 페이지의 읽은 줄 번호 (최대 {max_lines_in_page}줄)",
            min_value=1,
            max_value=max_lines_in_page,
            value=min(30, max_lines_in_page), 
            step=1,
            help=f"이 페이지는 총 {max_lines_in_page}줄로 이루어져 있습니다."
        )

# 4. 메인 화면: 채팅 인터페이스
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("소설 내용에 대해 질문해 보세요!"):
    if st.session_state.index is None:
        st.warning("먼저 왼쪽 사이드바에서 PDF 파일을 업로드해 주세요.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("스포일러를 제거한 전후 문맥을 분석 중입니다..."):
                
                # 사용자가 지정한 읽은 위치의 절대값 계산
                current_read_index = (current_volume * 100000000) + (current_page * 10000) + current_line
                
                # 1단계: 읽은 위치 이하의 소형 청크 3개 검색
                filters = MetadataFilters(
                    filters=[MetadataFilter(key="read_index", value=current_read_index, operator=FilterOperator.LTE)]
                )

                retriever = st.session_state.index.as_retriever(filters=filters, similarity_top_k=3)
                
                try:
                    retrieved_nodes = retriever.retrieve(prompt)
                    
                    if not retrieved_nodes:
                        bot_reply = "현재까지 읽으신 분량 내에서는 해당 질문에 대한 내용을 찾을 수 없습니다."
                        st.markdown(bot_reply)
                        st.session_state.messages.append({"role": "assistant", "content": bot_reply})
                    else:
                        # 2단계: 각 검색된 노드별로 앞뒤 약 3페이지(80줄) 문맥을 추출하고 스포일러는 철저히 잘라냄
                        safe_context_blocks = []
                        total_lines = len(st.session_state.all_lines)
                        
                        for node in retrieved_nodes:
                            start_idx = node.metadata["array_start_idx"]
                            
                            # 배열에서 검색할 앞뒤 인덱스 계산 (한 페이지 약 25줄 가정, 앞뒤 80줄 = 약 3페이지)
                            window_start = max(0, start_idx - 80)
                            window_end = min(total_lines, start_idx + 80)
                            
                            safe_lines = []
                            for i in range(window_start, window_end):
                                line_data = st.session_state.all_lines[i]
                                
                                # 🌟 [핵심 절대 방어 로직] 사용자가 읽은 부분을 초과하는 줄은 문맥에서 완전히 삭제!
                                if line_data["read_index"] <= current_read_index:
                                    # 문맥의 흐름을 돕기 위해 권/페이지 정보를 추가하여 텍스트 결합
                                    safe_lines.append(f"[{line_data['vol']}권 {line_data['page']}p] {line_data['text']}")
                            
                            if safe_lines:
                                safe_context_blocks.append("\n".join(safe_lines))
                        
                        # 3단계: 추출된 안전한 문맥들을 하나의 프롬프트로 결합하여 LLM에 직접 전달
                        final_context_str = "\n\n--- (다른 관련 장면) ---\n\n".join(safe_context_blocks)
                        
                        llm_prompt = f"""당신은 소설의 내용을 정확히 알려주는 AI 조수입니다. 아래 제공된 [소설 문맥]을 주의 깊게 읽고 질문에 답하세요. 문맥에는 사용자가 아직 읽지 않은 내용은 모두 제거되어 있으므로, 오직 제공된 문맥 안에서만 유추하여 답변해야 합니다.
[소설 문맥]
{final_context_str}

사용자 질문: {prompt}"""

                        # LLM에 직접 응답 요청 (복잡한 신시사이저 우회하여 속도 및 정확도 극대화)
                        response = Settings.llm.complete(llm_prompt)
                        
                        st.markdown(response.text)
                        st.session_state.messages.append({"role": "assistant", "content": response.text})
                        
                except Exception as e:
                    error_msg = str(e)
                    if "503" in error_msg or "UNAVAILABLE" in error_msg:
                        st.error("🚨 구글 AI 서버 과부하가 발생했습니다. 잠시 후 다시 시도해 주세요.")
                    elif "429" in error_msg or "Quota" in error_msg:
                        st.error("🚨 구글 API 무료 할당량을 초과했습니다. 잠시 후 다시 시도해 주세요.")
                    else:
                        st.error(f"예기치 못한 오류가 발생했습니다: {error_msg}")