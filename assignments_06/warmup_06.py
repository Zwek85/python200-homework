
from dotenv import load_dotenv
import os

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

# --- RAG Concepts ---
# Concepts Q1
# Scenario A: RAG
#   The policy library is large and changes every quarter, so retrieving the
#   relevant passages at query time keeps answers current without retraining.
#   Updating the index is far cheaper than fine-tuning on new PDFs each quarter.
#
# Scenario B: Fine-tuning
#   A dry, minimalist brand voice is a style the base model has seen little of,
#   and the team has 3,000 examples to train on. Fine-tuning bakes the style
#   into the model, which prompting alone can't do reliably.
#
# Scenario C: Prompt engineering
#   The report is only two pages, so it fits directly in the prompt as context.
#   Building a retrieval pipeline or training a model would be overkill for a
#   one-off document.

# Concepts Q2
# A confidently wrong answer is more harmful than "I am not sure" because the
# uncertainty signal is what tells a reader to double-check. "I am not sure"
# prompts verification; a confident wrong answer removes that prompt, so the
# error gets acted on or passed along unchallenged.
#
# Tone matters because people use fluent, assured language as a shortcut for
# reliability. A hallucination is worded just like a correct answer: same
# polished style, same authority, no hedging. Content and tone together make
# it hard to spot, since nothing in how it reads warns the reader it is wrong.
#
# Example: a nurse asks an AI assistant for the maximum safe dose of a drug
# and it confidently states a number that is too high. Because the answer
# sounds authoritative, the nurse may not check it against a drug reference,
# and a patient could be given a dangerous dose.

# Concepts Q3
# Original (out of order) list:
# steps = [
#     "Generate a response from the LLM",
#     "Extract text from source documents",
#     "Receive the user's query",
#     "Retrieve the most relevant chunks",
#     "Convert text chunks into embeddings",
#     "Inject retrieved chunks into the prompt",
#     "Split text into chunks",
#     "Embed the user's query",
# ]
#
# Correct order:
# 1. Extract text from source documents
#    Raw text is pulled out of the source files (PDFs, docs, etc.) so it can be processed.
# 2. Split text into chunks
#    The extracted text is divided into smaller pieces so each one fits in the model's
#    context window and covers a focused idea.
# 3. Convert text chunks into embeddings
#    Each chunk is turned into a numeric vector that captures its meaning, and the
#    vectors are stored in an index.
# 4. Receive the user's query
#    The user asks a question, which starts the retrieval part of the pipeline.
# 5. Embed the user's query
#    The question is converted into a vector using the same embedding model as the chunks.
# 6. Retrieve the most relevant chunks
#    The query vector is compared to the chunk vectors, and the closest matches are returned.
# 7. Inject retrieved chunks into the prompt
#    The retrieved text is added to the prompt as context alongside the user's question.
# 8. Generate a response from the LLM
#    The LLM writes an answer grounded in the supplied context.

# --- Keyword RAG ---
import string

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]

# Keyword Q1
query = "What are your hours on weekends?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

result = simple_keyword_retrieval(query, documents, verbose=True)
print(f"Selected document: {result[0][0]}")

# Keyword Q1 explanation:
# The function selected loyalty.txt, which is the wrong document. The filtered
# query tokens were ['hours', 'weekends', 'what', 'your']. hours.txt matched only
# "weekends" (1), while hiring.txt and loyalty.txt each matched only "your" (1),
# so three documents tied with a score of 1. The word "hours" never appears in
# hours.txt (it says "open" and "close" instead), and "your" is not in the
# stopword list, so it counted as a match. Because scores.sort(reverse=True)
# breaks ties by comparing the document names, "loyalty.txt" sorted first
# alphabetically in reverse and won. The correct answer (hours.txt) lost only
# because of the tie-break, not because the function judged it a better match.

# Keyword Q2
query = "Do you have anything without caffeine?"

result = simple_keyword_retrieval(query, documents, verbose=True)
print(f"Selected document: {result[0][0]}")

# Keyword Q2 explanation:
# No document was selected: the function returned "None found". The filtered
# query tokens were ['anything', 'caffeine', 'do', 'have', 'without'], and none
# of them appear in any of the four documents (overlap=0 for all).
# Keyword RAG got this wrong. menu.txt is clearly the relevant document, since
# it lists oat milk, almond milk, and other options a caffeine-free customer
# would care about, but it never uses the word "caffeine" or "without".
# Keyword matching only sees shared words, so it can't connect "without
# caffeine" to decaf-friendly or non-coffee drinks. Semantic retrieval (vector
# embeddings) would do better, because it compares meaning rather than exact
# words and would place this question close to the menu text about drinks.

# Keyword Q3 prediction (written before running the code):
# I predict the function will return "None found". The filtered query tokens
# should be ['do', 'how', 'i', 'rewards', 'sign', 'up'] ("for" is a stopword).
# The right document is loyalty.txt, but it says "loyalty program" and "points"
# and never uses the word "rewards" or "sign up", so I expect zero overlap with
# every document. "How", "do", "i", and "up" are not in the stopword list, but I
# don't see them in any of the documents either.

# Keyword Q3
query = "How do I sign up for rewards?"

result = simple_keyword_retrieval(query, documents, verbose=True)
print(f"Selected document: {result[0][0]}")

# Keyword Q3 result:
# My prediction was correct: the function returned "None found" because every
# document had overlap=0. The relevant document is loyalty.txt, but it says
# "Join our loyalty program" and "points", while the query says "sign up" and
# "rewards". Same meaning, different words, so keyword matching finds nothing.
# This is the same weakness as Q2: keyword RAG needs the exact words to match.

# --- Semantic RAG Concepts ---
# Semantic Q1
# 1. A vector embedding is a list of numbers that represents the meaning of a
#    piece of text, so that texts with similar meanings end up with similar
#    lists of numbers (close together in vector space).
#
# 2. The chunk with the 0.85 score is more relevant. Cosine similarity measures
#    how closely two vectors point in the same direction, so 0.85 means the
#    chunk and the query are very close in meaning, while 0.30 means they are
#    only loosely related.
#
# 3. Embeddings capture meaning rather than exact spelling, so a chunk that says
#    "join our loyalty program" lands near a query about "signing up for
#    rewards" even though they share no words. The model learned during
#    training that those phrases are used in similar contexts.

# Semantic Q2
# | Feature                    | Keyword RAG                       | Semantic RAG                                       |
# |----------------------------|-----------------------------------|----------------------------------------------------|
# | What is compared?          | Exact word overlap                | Meaning, by comparing query and chunk embeddings   |
# | What is retrieved?         | Full document                     | The top-k most similar chunks (passages)           |
# | Can it handle synonyms?    | No                                | Yes, similar meanings get similar embeddings       |
# | Storage format             | Plain text dictionary             | Vector index (embeddings stored with chunk text)   |
# | Relevance score            | Number of overlapping keywords    | Cosine similarity between the two vectors          |

# --- LlamaIndex ---
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from pathlib import Path


Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = OpenAI(model="gpt-4o-mini")

# Adjust this path to match where brightleaf_pdfs/ lives on your computer.
PDF_DIR = Path.home() / "Documents" / "python-200-v1" / "lessons" / "06_AI_augmentation" / "resources" / "brightleaf_pdfs"

pdf_documents = SimpleDirectoryReader(PDF_DIR).load_data()
index = VectorStoreIndex.from_documents(pdf_documents)


def show_result(question, response):
    """Print the question, the answer, and each retrieved source node."""
    print(f"\nQuestion: {question}")
    print(f"Answer: {response}")
    for i, node in enumerate(response.source_nodes, start=1):
        print(f"  Source {i} | score={node.score:.4f} | {node.node.get_content()[:150]!r}")


# LlamaIndex Q1
questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]

query_engine = index.as_query_engine(similarity_top_k=3)
for question in questions:
    response = query_engine.query(question)
    show_result(question, response)

# LlamaIndex Q1 observations (TODO: fill in after running, one note per query):
# Query 1 (employee benefits):
#   - Relevant chunks?
#   - Tone (confident and specific, or hedging)?
#   - Anything unexpected retrieved?
# Query 2 (security policies):
#   - Relevant chunks?
#   - Tone (confident and specific, or hedging)?
#   - Anything unexpected retrieved?

# LlamaIndex Q2
q2 = questions[0]
for k in (1, 5):
    print(f"\n=== similarity_top_k={k} ===")
    engine_k = index.as_query_engine(similarity_top_k=k)
    show_result(q2, engine_k.query(q2))

# LlamaIndex Q2 observations (TODO: fill in after running):
#   - How did the response change between top_k=1 and top_k=5?
#   - Is more retrieved context always better? (Think about: a larger k can add
#     missing details, but it can also add less relevant chunks that dilute or
#     distract from the answer, and it increases prompt size and cost.)

# LlamaIndex Q3
# Edit this query if you want to try a different kind of hard question.
q3 = "How does BrightLeaf's remote work policy compare to its travel expense rules?"

print("\n=== LlamaIndex Q3 ===")
response_q3 = query_engine.query(q3)
show_result(q3, response_q3)
print("\nAll retrieved chunks:")
for i, node in enumerate(response_q3.source_nodes, start=1):
    print(f"\n--- Chunk {i} (score={node.score:.4f}) ---")
    print(node.node.get_content())

# LlamaIndex Q3 observations (TODO: fill in after running):
#   - What I expected:
#   - What actually happened:
#   - What I would change (e.g. smaller or larger chunks, higher top_k, query
#     rewriting, metadata filters, a reranker, or telling the model to say
#     "I don't know" when the context doesn't contain the answer):

# LlamaIndex Q4
judge_llm = OpenAI(model="gpt-4o-mini")
faithfulness_evaluator = FaithfulnessEvaluator(llm=judge_llm)
relevancy_evaluator = RelevancyEvaluator(llm=judge_llm)


def evaluate(q):
    """Run both evaluators on the response to q and print the scores."""
    response = query_engine.query(q)
    faith = faithfulness_evaluator.evaluate_response(response=response)
    rel = relevancy_evaluator.evaluate_response(query=q, response=response)
    print(f"\nQuery: {q}")
    print(f"Response: {response}")
    print(f"Faithfulness score: {faith.score} (passing={faith.passing})")
    print(f"Relevancy score: {rel.score} (passing={rel.passing})")


q = "What employee benefits does BrightLeaf offer?"
evaluate(q)

# A question that is clearly not covered by the Brightleaf documents.
evaluate("What is the best recipe for chocolate chip cookies?")

# LlamaIndex Q4 answers:
# 1. A faithfulness score of 1.0 means every claim in the response is supported
#    by the retrieved context. A score of 0.0 means the response contains
#    information that the retrieved chunks do not support, i.e. it was made up.
# 2. Relevancy measures whether the response (and retrieved context) actually
#    answers the question that was asked. Faithfulness checks the answer
#    against the sources; relevancy checks the answer against the question. A
#    response can be faithful but irrelevant, or relevant but unfaithful.
# 3. Yes, the scores changed. For the benefits question, faithfulness and
#    relevancy were both 1.0 [use your numbers]: the retrieved chunks contained
#    the answer and the model stuck to them. For the cookie question,
#    [faithfulness / relevancy] dropped to 0.0 because [the retrieved chunks
#    had nothing about cookies, so the model either said it couldn't answer or
#    made something up]. This shows the two metrics catch different failures.
# 4. LLM-as-a-judge means using a second LLM to grade the output of the first,
#    following a rubric (for example, "is every claim supported by this
#    context?"). It is used for RAG because answers are free-form text, so there
#    is no single correct string to compare against, and a simple accuracy
#    metric can't tell whether a differently worded answer is still correct.