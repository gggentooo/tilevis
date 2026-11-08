# Google Colab에서 트레이스 생성하기

Google Colab 노트북에 아래 코드 블록들을 순서대로 붙여넣고 실행하면 시연에 사용 가능한 트레이스 파일을 생성 가능합니다.

### 1단계: 저장소 클론 및 패키지 설치
```python
# 1. 프로젝트 저장소 클론 (또는 Colab 작업 디렉토리에 업로드)
!git clone https://github.com/your-username/llm_tile_visualizer.git
%cd llm_tile_visualizer

# 2. 모델 레코딩 전용 의존성(PyTorch, tiktoken)과 함께 tilevis 설치
!pip install -q -e ".[record]"
```

### 2단계: 체크포인트(`*.pth`) 업로드 또는 구글 드라이브 마운트
모델 가중치 파일이 구글 드라이브에 있다면 드라이브를 마운트합니다:
```python
from google.colab import drive
drive.mount('/content/drive')
# 예시 경로: /content/drive/MyDrive/my_model.pth
```
또는 Colab 좌측 파일 탐색기 창을 통해 직접 업로드할 수도 있습니다.

### 3단계: 모델 프로필 파일 작성 (`profiles/colab_model.toml`)
팀원의 체크포인트 경로와 모델 설정에 맞게 프로필 TOML을 작성합니다:
```python
profile_content = """
[model]
name = "team_model"
adapter = "gpt_scratch"  # previous_chapters.py 기반 GPT 모델
code_path = "ch07/01_main-chapter-code/previous_chapters.py"
checkpoint = "/content/drive/MyDrive/my_model.pth"  # .pth 파일 경로
device = "cuda"  # GPU 사용 (CPU일 경우 "cpu")

[model.overrides]
# state_dict에 저장되지 않은 하이퍼파라미터가 있다면 입력:
# n_layers = 12
# emb_dim = 768
# n_heads = 12
"""

with open("profiles/colab_model.toml", "w", encoding="utf-8") as f:
    f.write(profile_content)
```

### 4단계: 모델 구조 검증 (`inspect`)
가중치 텐서 형상과 파라미터가 정상적으로 로드되는지 확인합니다:
```python
!tilevis inspect --profile profiles/colab_model.toml
```

### 5단계: 순방향 패스 트레이스 파일 기록 (`record`)
원하는 프롬프트와 생성할 토큰 수를 지정하여 `.trace.npz` 파일을 생성합니다:
```python
!tilevis record \
  --profile profiles/colab_model.toml \
  --prompt "Every effort moves you forward" \
  --new-tokens 3 \
  --output my_model.trace.npz
```

### 6단계: 무결성 검사 및 로컬 다운로드
생성된 트레이스 파일의 구조를 사전 검증하고 로컬 컴퓨터로 다운로드합니다:
```python
# 1. 트레이스 파일 무결성 및 구조 검사
!tilevis check my_model.trace.npz

# 2. 발표용 로컬 PC로 다운로드
from google.colab import files
files.download("my_model.trace.npz")
```

---

### Key Notes / 참고 사항
- **Playback on Colab**: Colab은 GUI 디스플레이 창이 없는 원격 가상환경이므로 인터랙티브 창을 띄우는 `tilevis play`는 불가능하지만, `--headless` 모드(`!tilevis play my_model.trace.npz --headless`)를 통한 오프스크린 검증은 Colab에서도 동작합니다.
- **Portability**: 생성된 `my_model.trace.npz` 파일은 PyTorch나 GPU 없이도 발표장의 Windows PC(`tilevis play` 또는 `play_demo.bat`)에서 즉시 부드럽게 재생됩니다.