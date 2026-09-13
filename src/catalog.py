"""dresses 카탈로그 임베딩 생성/캐싱과 Top-K 검색.

이 과제는 이미지-이미지 검색이라, 검색/랭킹은 전부 **독립적인 image_id 단위**로 처리한다.
카탈로그는 item_id 32,434개지만 고유 이미지는 28,555장뿐인데(같은 사진을 여러 item_id/geo가
공유), item_id 단위로 Top-k 슬롯을 펼쳐 담는 건 이미지 검색 문제 자체와는 무관한 카탈로그
메타데이터 중복일 뿐이라 의미가 없다고 판단 -> Top-k는 항상 "서로 다른 이미지 k장"이고,
그 이미지에 딸린 item_id들은 슬롯을 추가로 차지하지 않고 참고 정보로만 같이 반환한다.

28,555장 중에서도 근접중복(diff<5, 사실상 동일 사진이지만 image_id는 다름)이 있음
(dedup_catalog_pool.py --scope full로 확인, eval/results/full_pool_dedup.json). 그대로 두면
Top-10 슬롯 여러 개가 눈으로 구분 안 되는 같은 사진으로 낭비되므로, 검색 후보 자체를 대표
이미지로 축소해서 애초에 같은 사진이 여러 슬롯을 차지하지 못하게 한다.
"""
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch

from config import CATALOG_ITEMS_CSV, CATALOG_IMAGES_CSV, CATALOG_IMAGES_DIR, EMBED_CACHE_DIR, PROJECT_ROOT

FULL_POOL_DEDUP_PATH = PROJECT_ROOT / "eval" / "results" / "full_pool_dedup.json"


def load_catalog_tables():
    items = pd.read_csv(CATALOG_ITEMS_CSV)
    images = pd.read_csv(CATALOG_IMAGES_CSV)
    return items, images


def load_full_pool_dedup():
    """dedup_catalog_pool.py --scope full 산출물. 없으면 (None, None) 반환 (dedup 건너뜀)."""
    if not FULL_POOL_DEDUP_PATH.exists():
        return None, None
    raw = json.loads(FULL_POOL_DEDUP_PATH.read_text(encoding="utf-8"))
    representative_of = {int(k): int(v) for k, v in raw["representative_of"].items()}
    representatives = set(int(x) for x in raw["representatives"])
    return representative_of, representatives


def dedup_catalog_embeddings(image_ids: List[int], embeddings: torch.Tensor, representatives: set):
    """검색 후보를 대표 이미지만으로 축소 (근접중복 이미지가 Top-K 슬롯을 중복 낭비하지 않도록)."""
    keep_idx = [i for i, iid in enumerate(image_ids) if iid in representatives]
    kept_ids = [image_ids[i] for i in keep_idx]
    kept_emb = embeddings[torch.tensor(keep_idx)]
    return kept_ids, kept_emb


def build_or_load_catalog_embeddings(encoder, cache_name: str, force: bool = False, batch_size: int = 32):
    """cache_name 예: 'siglip2_pretrained'. 반환: (image_ids: List[int], embeddings: (N,D) tensor)."""
    cache_path = EMBED_CACHE_DIR / f"{cache_name}_catalog.pt"
    _, images = load_catalog_tables()

    if cache_path.exists() and not force:
        cached = torch.load(cache_path, weights_only=False)
        if list(cached["image_ids"]) == list(images["image_id"]):
            return cached["image_ids"], cached["embeddings"]
        # 카탈로그 구성이 바뀐 경우에만 재계산
        print("catalog composition changed, recomputing embeddings")

    image_ids = images["image_id"].tolist()
    paths = [CATALOG_IMAGES_DIR / f"{iid}.jpg" for iid in image_ids]
    embeddings = encoder.encode_images(paths, batch_size=batch_size)
    torch.save({"image_ids": image_ids, "embeddings": embeddings}, cache_path)
    print(f"saved catalog embeddings -> {cache_path} ({embeddings.shape})")
    return image_ids, embeddings


def search_topk_items(query_embedding: torch.Tensor, image_ids: List[int], catalog_embeddings: torch.Tensor,
                       items_df: pd.DataFrame, k: int = 10, representative_of: Optional[dict] = None):
    """query_embedding: (D,) 정규화된 벡터. 반환: **이미지 단위** top-k 결과 리스트[dict], 정확히 k개
    (또는 후보가 k개 미만이면 그만큼)의 서로 다른 image_id.

    이미지-이미지 검색 과제라 item_id 단위로 슬롯을 펼치지 않는다 - 같은 이미지를 쓰는 item_id들은
    'item_ids' 필드에 참고용으로 전부 담되, 그것 때문에 다른 이미지가 Top-k에서 밀려나지 않는다.
    'item_id' 필드는 하위 호환/표시 편의용으로 그 이미지의 item_id 중 대표 1개(첫 번째)를 담는다.

    representative_of가 주어지면(근접중복 클러스터 매핑), 원래 다른 image_id를 쓰던 item이라도
    같은 클러스터의 대표 이미지 자리에서 item_ids에 합쳐져 표시된다 (item이 결과에서 안 보이게
    누락되는 일 없이, 그렇다고 같은 사진이 여러 슬롯을 차지하지도 않도록).
    """
    sims = catalog_embeddings @ query_embedding  # (N,), cosine sim (둘 다 L2-normalize됨)
    order = torch.argsort(sims, descending=True)[:k]

    items_df = items_df.copy()
    if representative_of:
        items_df["image_id"] = items_df["image_id"].map(lambda x: representative_of.get(int(x), int(x)))

    img_to_items = items_df.groupby("image_id")["item_id"].apply(list).to_dict()
    img_to_name = items_df.drop_duplicates("image_id").set_index("image_id")["name"].to_dict()

    results = []
    for rank, idx in enumerate(order.tolist(), start=1):
        img_id = image_ids[idx]
        item_ids = img_to_items.get(img_id, [])
        results.append({
            "rank": rank,
            "image_id": img_id,
            "score": sims[idx].item(),
            "item_ids": item_ids,
            "item_id": item_ids[0] if item_ids else None,
            "name": img_to_name.get(img_id, ""),
        })
    return results
