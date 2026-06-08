import asyncio

from dotenv import load_dotenv
load_dotenv()

from chromadb import PersistentClient
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.llms.openai import OpenAI
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Document
from llama_index.core.utils import iter_batch
from tqdm.asyncio import tqdm

from src.modules.datasets.ex_fever import DocDB
from src.modules.property_graph.indices.hippo_rag import (
    HippoRAGGraphIndex,
)

NUM_WORKERS = 4


async def process_batch(semaphore: asyncio.Semaphore, doc_batch: list, index: HippoRAGGraphIndex, docdb: DocDB) -> None:
    async with semaphore:
        documents = []
        for doc_id in doc_batch:
            doc_text = docdb.get_doc_text(doc_id)
            if doc_text:
                documents.append(Document(
                    text=doc_text,
                    metadata={"doc_id": doc_id},
                ))
        if documents:
            await asyncio.to_thread(index.insert_nodes, documents)


async def main() -> None:
    llm = OpenAI(model='gpt-4.1-mini', temperature=0)
    # llm = AzureOpenAI(
    #     model="gpt-4.1-mini",
    #     deployment_name="gpt-4.1-mini",
    #     temperature=0
    # )
    embed = OpenAIEmbedding(model="text-embedding-3-small")
    docdb = DocDB("datas/ex-fever/wiki_db.db")
    doc_ids = docdb.get_doc_ids()

    vector_db = PersistentClient(path="output/ex-fever/chroma")
    vector_collection = vector_db.get_or_create_collection("ex-fever")
    vector_store = ChromaVectorStore(chroma_collection=vector_collection)
    graph_store = Neo4jPropertyGraphStore(
        username="username",
        password="pw",
        url="url",
        database="db",
        refresh_schema=False
    )

    START_IDX = 4200
    PROCESS_BATCH = 50

    batches = list(iter_batch(doc_ids[START_IDX:], PROCESS_BATCH))

    if START_IDX != 0:
        index = HippoRAGGraphIndex.from_existing(
            property_graph_store=graph_store,
            vector_store=vector_store,
            llm=llm,
            embed_model=embed,
        )
    else:
        first_batch = batches.pop(0)
        first_docs = []
        for doc_id in first_batch:
            doc_text = docdb.get_doc_text(doc_id)
            if doc_text:
                first_docs.append(Document(
                    text=doc_text,
                    metadata={"doc_id": doc_id},
                ))
        index = await asyncio.to_thread(
            HippoRAGGraphIndex,
            nodes=first_docs,
            llm=llm,
            embed_model=embed,
            vector_store=vector_store,
            property_graph_store=graph_store,
        )

    semaphore = asyncio.Semaphore(NUM_WORKERS)
    tasks = [process_batch(semaphore, batch, index, docdb) for batch in batches]
    await tqdm.gather(*tasks, desc="Indexing", unit="batch")


if __name__ == '__main__':
    asyncio.run(main())
