"""프로젝트 공통 경로/상수. 노트북과 스크립트 모두 여기서 import해서 쓴다.

제출용 버전: 원본 프로젝트는 절대경로(C:\\jupyter\\레브잇_과제)를 썼지만, 이 폴더는 어느 위치에
두어도 동작하도록 PROJECT_ROOT를 이 파일 위치(src/) 기준 상대경로로 계산한다. 학습/baseline 비교
등 이 재현 노트북(추론+시각화)에 필요 없는 코드는 제외했으므로, 여기 상수도 실제 쓰이는 것만 남겼다.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # src/ 의 부모 = 프로젝트 루트

# 데이터
CATALOG_DIR = PROJECT_ROOT / "data" / "glami1m_dresses"
CATALOG_ITEMS_CSV = CATALOG_DIR / "catalog_items.csv"      # item_id 기준 32,434행 (최종 카탈로그 단위)
CATALOG_IMAGES_CSV = CATALOG_DIR / "catalog_images.csv"    # image_id 기준 28,555행 (임베딩 계산 단위)
CATALOG_IMAGES_DIR = CATALOG_DIR / "images"                # <image_id>.jpg (GLAMI-1M에서 별도 다운로드 필요)

# 모델
SIGLIP2_HF_ID = "google/siglip2-so400m-patch14-384"
# 원본 개발 환경에서는 HF Hub 파이썬 클라이언트가 자주 멈춰서 curl로 받은 로컬 사본을 썼지만,
# 이 폴더에는 그 로컬 사본을 포함하지 않는다. 로컬 사본이 없으면 HF Hub ID로 바로 로드한다
# (transformers.from_pretrained()는 로컬 경로/Hub ID 둘 다 동일하게 처리한다).
_SIGLIP2_LOCAL_DIR = PROJECT_ROOT / "models" / "siglip2-so400m-patch14-384-local"
SIGLIP2_LOCAL_HF_DIR = _SIGLIP2_LOCAL_DIR if _SIGLIP2_LOCAL_DIR.exists() else SIGLIP2_HF_ID

EMBED_CACHE_DIR = PROJECT_ROOT / "eval" / "embed_cache"
EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda"
