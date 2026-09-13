"""LoRA fine-tuned SigLIP2 vision encoder 래퍼.

- src/train_lora.py로 학습한 peft LoRA 어댑터(models/checkpoints/<run_name>/step_XXXX/)를
  베이스 SigLIP2 vision tower(SIGLIP2_LOCAL_HF_DIR)에 얹어서 로드한다.
- Siglip2Encoder와 동일한 인터페이스(encode_images -> L2-normalize (N,D) cpu float32 tensor)라
  catalog.py/search_query.py를 그대로 재사용할 수 있다.
- 체크포인트마다 검색 품질이 다를 수 있어(아이디어_진행.txt 참고: 100 step 근방이 좋고 이후 저하되는
  경향 관찰됨) adapter_dir만 바꿔서 여러 체크포인트를 쉽게 비교할 수 있게 만들었다.
"""
from pathlib import Path
from typing import List, Union

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoImageProcessor, SiglipVisionModel

from config import SIGLIP2_LOCAL_HF_DIR, DEVICE


class Siglip2LoraEncoder:
    def __init__(self, adapter_dir: Union[str, Path], device: str = DEVICE, dtype=torch.float16):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.dtype = dtype if self.device == "cuda" else torch.float32

        self.processor = AutoImageProcessor.from_pretrained(str(SIGLIP2_LOCAL_HF_DIR))
        base_model = SiglipVisionModel.from_pretrained(str(SIGLIP2_LOCAL_HF_DIR), dtype=self.dtype)
        peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        # 추론만 할 거라 LoRA 저랭크 행렬을 베이스 가중치에 합쳐버린다 (merge_and_unload).
        # 어댑터를 분리해둔 채로 매 forward마다 추가 행렬곱을 하는 것보다 훨씬 빠르다
        # (27개 레이어 x attention 4곳 = 108군데의 추가 연산+Python 오버헤드가 카탈로그 전체 인코딩에서
        # 누적되면 순정 SiglipVisionModel보다 오히려 느려지는 걸 실측으로 확인함).
        self.model = peft_model.merge_and_unload()
        self.model.to(self.device).eval()

    @torch.no_grad()
    def encode_images(self, images: List[Union[str, Path, Image.Image]], batch_size: int = 32) -> torch.Tensor:
        """이미지 리스트 -> L2-normalize된 (N, D) 임베딩 텐서 (cpu, float32)."""
        pil_images = [self._load(im) for im in images]
        embeds = []
        for i in range(0, len(pil_images), batch_size):
            batch = pil_images[i : i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device, dtype=self.dtype)
            out = self.model(**inputs)
            pooled = out.pooler_output
            pooled = torch.nn.functional.normalize(pooled.float(), dim=-1)
            embeds.append(pooled.cpu())
        return torch.cat(embeds, dim=0)

    @staticmethod
    def _load(im: Union[str, Path, Image.Image]) -> Image.Image:
        if isinstance(im, Image.Image):
            return im.convert("RGB")
        return Image.open(im).convert("RGB")
