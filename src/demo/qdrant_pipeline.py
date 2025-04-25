import asyncio
from pathlib import Path
from uuid import uuid4
from typing import Any

import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModel

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct

import logging

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================== Config ==================
collection_name = "speculative_rag"
dimension = 1024  # InBedderRoberta 임베딩 차원
data_rows = 3000 # 원래 50_000

# 모델 경로
embedding_model = "KomeijiForce/inbedder-roberta-large"
tokenizer = AutoTokenizer.from_pretrained(embedding_model)
model = AutoModel.from_pretrained(embedding_model)
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model.eval().cuda()

# Qdrant 클라이언트
qdrant_client = AsyncQdrantClient(path=Path("qdrant_client"))

# ================== Helper ==================
def get_embedding(text: str, max_length: int = 512) -> list[float]:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding="max_length",
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        masked = last_hidden_state * attention_mask
        summed = masked.sum(1)
        counted = attention_mask.sum(1)
        mean_pooled = summed / counted

    return mean_pooled[0].cpu().tolist()

async def create_point(example: dict[str, Any], tokenizer, model) -> models.PointStruct:
    text = example.get("content", "")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()} 
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

    references = example.get("references")
    if isinstance(references, np.ndarray):
        references = references.tolist()
    elif references is None:
        references = []

    return models.PointStruct(
        id=str(uuid4()),
        vector=embedding,
        payload=dict(
            chunk_id=example.get("id"),
            arxiv_id=example.get("arxiv_id"),
            title=example.get("title"),
            content=example.get("content"),
            prechunk_id=example.get("prechunk_id"),
            postchunk_id=example.get("postchunk_id"),
            references=references,
        ),
    )


async def process_batch(batch: list[dict[str, Any]], tokenizer, model) -> list[PointStruct]:
    return await asyncio.gather(*[create_point(example, tokenizer, model) for example in batch])


# ================== Main ==================
async def main():
    # Qdrant collection 생성 여부 확인
    current_collections = await qdrant_client.get_collections()
    if collection_name not in [col.name for col in current_collections.collections]:
        logger.info("Collection '%s' doesn't exist. Creating...", collection_name)
        await qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dimension, distance=Distance.DOT),
        )
        logger.info("Collection '%s' created!", collection_name)
    else:
        logger.info("Collection '%s' already exists. Skipping creation.", collection_name)

    # 데이터셋 로드
    dataset: Dataset = load_dataset("jamescalam/ai-arxiv2-semantic-chunks", split="train")
    df = dataset.to_pandas().iloc[:data_rows]
    records = df.to_dict(orient="records")

    # 배치 단위로 업로드
    batch_size = 128
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        points = await process_batch(batch, tokenizer, model)
        await qdrant_client.upsert(collection_name=collection_name, points=points)
        logger.info("Upserted batch %d/%d", i + batch_size, len(records))

if __name__ == "__main__":
    asyncio.run(main())
