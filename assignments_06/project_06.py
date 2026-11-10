import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI

# --- Step 1: Setup ---
load_dotenv()  # finds the root .env by searching upward from this file

api_key = os.getenv("OPENAI_API_KEY")
assert api_key, "OPENAI_API_KEY not found. Check your .env file."
client = OpenAI(api_key=api_key)
print("API key loaded successfully.")

docs_dir = Path(__file__).parent / "resources" / "groundwork_docs"
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"
print(f"Found documents directory: {docs_dir}")

# --- Step 2: Load the documents ---
documents = SimpleDirectoryReader(input_dir=str(docs_dir)).load_data()

print(f"Loaded {len(documents)} documents:")
for doc in documents:
    print(f"  {doc.metadata['file_name']}")

# --- Step 3: Build the index and query engine ---
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = LlamaOpenAI(model="gpt-4o-mini")

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=3)

print("Index built successfully. Ready to answer questions.")

# --- Step 4: Query the assistant ---
questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

for question in questions:
    response = query_engine.query(question)
    top_node = response.source_nodes[0]

    print("=" * 70)
    print(f"Question: {question}")
    print(f"Answer: {response}")
    print("Top source:")
    print(f"  Document: {top_node.node.metadata['file_name']}")
    print(f"  Similarity score: {top_node.score:.4f}")
    print(f"  Chunk preview: {top_node.node.get_content()[:200]}")

# Reflection on Step 4:
# [Write this from your own output: did the answers sound confident and
# accurate? Did any surprise you, for example the loyalty program question?]

# --- Step 5: Find a failure ---
failure_question = "Do you have a drive-through, and can I order ahead on your mobile app?"

response = query_engine.query(failure_question)

print("=" * 70)
print(f"Question: {failure_question}")
print(f"Full response: {response}")
print("\nRetrieved source nodes:")
for i, node in enumerate(response.source_nodes, start=1):
    print(f"  [{i}] Document: {node.node.metadata['file_name']}")
    print(f"      Similarity score: {node.score:.4f}")
    print(f"      Chunk preview: {node.node.get_content()[:200]}")

# Failure analysis:
# What I asked / why it was hard: I asked whether Groundwork has a drive-through
# and mobile ordering, because none of the five documents mention either, so the
# right answer is "the documents don't say."
#
# What went wrong: Retrieval returned three chunks (faq.txt, wholesale_catering.txt,
# seasonal_specials.txt) with low similarity scores (0.33, 0.30, 0.27), none of
# which actually address the question. The retriever always returns top_k chunks
# even when nothing is relevant. The model, however, did not guess: it said no
# information was provided.
#
# Tone: The model's tone was plain and uncertain rather than confident, which is
# the correct behavior. But this depends on the model choosing to admit
# ignorance; nothing in the system enforces it. A different question or model
# could have produced a confident guess from those irrelevant chunks. This
# suggests AI answers should be checked against their sources, and that low
# similarity scores are a warning sign worth surfacing.
#
# What I would change: add a similarity cutoff so low-scoring chunks are dropped
# and the system returns a standard "I don't have that information" answer, add
# a system prompt telling the model to answer only from the retrieved context,
# and add documents covering gaps like ordering and services.

# --- Step 6: Reflection ---
#
# 1. Framework value:
# The core LlamaIndex pipeline took about 5 lines: SimpleDirectoryReader to load
# the documents, two Settings lines for the embedding model and LLM,
# VectorStoreIndex.from_documents to chunk, embed, and index, and as_query_engine
# to set up retrieval and answer generation. Doing chunking, embedding, and indexing
# myself would mean writing a text splitter, calling the embeddings API in
# batches, storing the vectors, computing cosine similarity, ranking results,
# and building the prompt that combines the question with retrieved text. A
# framework hides all of that behind a few calls, which makes it fast to build a
# working prototype. The tradeoff is that defaults like chunk size, the fallback
# LLM (which caused my 404 error), and "always return top_k chunks" happen out
# of sight, so I have to understand them to debug or improve the system.
#
# 2. Another use case:
# A company's internal HR or IT help desk. Employees constantly ask questions
# like "How many vacation days do I get?" or "How do I set up the VPN?", and the
# answers are in long policy documents and wikis. A RAG assistant could answer
# from the current documents and show which policy each answer came from, so
# staff can verify it, saving the HR/IT team from answering the same questions
# repeatedly.
#
# 3. A failure mode RAG cannot fully prevent:
# The model can still misread or misstate correct information even when
# retrieval works, for example by merging details from two chunks into a wrong
# conclusion. Retrieval only supplies the right text; the LLM still generates
# the answer, and generated text can be fluent, confident, and wrong. The
# documents themselves can also be outdated or wrong, and RAG will faithfully
# repeat bad source material. Important answers still need source checking or
# human review.