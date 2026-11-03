# --- Completions API ---
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

# API Q1
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "What is one thing that makes Python a good language for beginners?",
        }
    ],
)

print("API Q1 - Response text:", response.choices[0].message.content)
print("API Q1 - Model:", response.model)
print("API Q1 - Total tokens used:", response.usage.total_tokens)

# API Q2
prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

for temp in temperatures:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temp,
    )
    print(f"API Q2 - Temperature {temp}: {response.choices[0].message.content}")

# What I notice: 
#Temperature 0 and 0.7 both gave "DataForge Solutions", but the
# 0 response included an explanation while 0.7 gave only the name. At 1.5 the
# model chose a different name ("DataCraft Studio"). With one run each the
# differences were small, but higher temperature allows less likely word
# choices, so outputs vary more from run to run.
# For consistent, reproducible output I would use temperature 0, since it
# always favors the most likely tokens (though OpenAI doesn't guarantee
# perfectly identical results every time).
# Which temperature for consistent, reproducible output? 0

# API Q3
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Give me a one-sentence fun fact about pandas (the animal, not the library).",
        }
    ],
    n=3,
    temperature=1.0,
)

for i, choice in enumerate(response.choices, start=1):
    print(f"API Q3 - Completion {i}: {choice.message.content}")

# API Q4
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain how neural networks work."}],
    max_tokens=15,
)

print("API Q4 - Response text:", response.choices[0].message.content)
print("API Q4 - Finish reason:", response.choices[0].finish_reason)
print("API Q4 - Completion tokens:", response.usage.completion_tokens)

# What happened: The response was cut off mid-sentence ("...inspired by the way
# biological neural networks in the") after exactly 15 tokens. The finish reason
# was "length" instead of "stop", which means the model hit the token limit
# before it finished. max_tokens doesn't make the model write a shorter answer,
# it just stops it at the limit.
# Why use max_tokens in a real application? To control cost (you pay per token),
# keep response times short, prevent unexpectedly long outputs, and make sure
# responses fit limited space such as a preview, a notification, or a UI card.

# --- System Messages ---

# System Q1
question = "I don't understand what a list comprehension is."

# Personality 1: patient tutor
messages = [
    {
        "role": "system",
        "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement.",
    },
    {"role": "user", "content": question},
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
)
print("System Q1 - Patient tutor:", response.choices[0].message.content)

# Personality 2: different system message, same question
messages = [
    {
        "role": "system",
        "content": "You are a grumpy old pirate captain who explains everything using sea and ship metaphors, and talks like a pirate.",
    },
    {"role": "user", "content": question},
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
)
print("System Q1 - Pirate captain:", response.choices[0].message.content)

# What changed: Only the system message changed, and the same question got a
# very different style of answer. The patient tutor gave a plain, step-by-step
# explanation with a before/after code example, then ended with encouragement
# ("keep practicing... you're doing great"), exactly as its system message
# instructed. The pirate captain used pirate slang and nautical metaphors
# (a crew gathering treasure) and gave a shorter, less detailed example.
# Both were still accurate about what a list comprehension is, so the system
# message changed the tone, framing, and level of detail rather than the facts.

# System Q2
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {
        "role": "assistant",
        "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?",
    },
    {"role": "user", "content": "Can you remind me what my name is?"},
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
)
print("System Q2 - Response:", response.choices[0].message.content)

# Why does the model know Jordan's name? The API is stateless, so the model
# doesn't remember anything from earlier calls. It knew the name because I sent
# the whole conversation history in the messages list, including the earlier
# user message ("My name is Jordan...") and the assistant reply. The model reads
# that entire list on every call and answers the last message using it as
# context. The "memory" comes from the history I pass in, not from the model.

# --- Prompt Engineering ---

# Prompt Q1 - Zero-Shot
reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow.",
]

for i, review in enumerate(reviews, start=1):
    prompt = (
        "Classify the sentiment of the following review as positive, negative, "
        "or mixed. Respond with only one word.\n\n"
        f"Review: {review}"
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    print(f"Prompt Q1 - Review {i}: {response.choices[0].message.content}")

# Prompt Q2 - One-Shot
for i, review in enumerate(reviews, start=1):
    prompt = (
        "Classify the sentiment of the following review as positive, negative, "
        "or mixed.\n\n"
        "Example:\n"
        'Review: "Fast shipping but the item arrived damaged."\n'
        "Sentiment: mixed\n\n"
        f'Review: "{review}"\n'
        "Sentiment:"
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    print(f"Prompt Q2 - Review {i}: {response.choices[0].message.content}")

# Did adding one example change the format or consistency? Yes, it changed the
# format. In Q1 the model labeled review 3 as "Mixed" (capitalized), because I
# only described the task. In Q2 all three labels came back lowercase
# (positive, negative, mixed), matching the format of the example
# ("Sentiment: mixed"). The classifications themselves were correct in both
# cases, so the example mainly improved format consistency rather than
# accuracy. Showing the model what I want works well for controlling output
# format.

# Prompt Q3 - Few-Shot
for i, review in enumerate(reviews, start=1):
    prompt = (
        "Classify the sentiment of the following review as positive, negative, "
        "or mixed.\n\n"
        "Example 1:\n"
        'Review: "The interface is clean and everything works exactly as promised."\n'
        "Sentiment: positive\n\n"
        "Example 2:\n"
        'Review: "I waited two weeks for a refund and nobody ever replied."\n'
        "Sentiment: negative\n\n"
        "Example 3:\n"
        'Review: "Fast shipping but the item arrived damaged."\n'
        "Sentiment: mixed\n\n"
        f'Review: "{review}"\n'
        "Sentiment:"
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    print(f"Prompt Q3 - Review {i}: {response.choices[0].message.content}")

# Comparing the three approaches: Zero-shot (Q1) gave correct classifications
# but the format was inconsistent with what I wanted (for example, "Mixed"
# was capitalized), since I only described the task. One-shot (Q2) fixed the
# format: all labels came back lowercase, matching the example. Few-shot (Q3)
# gave the same correct, consistently formatted results as one-shot. For this
# simple task, the extra examples didn't change the output.
#
# When would I choose each one?
# - Zero-shot: for simple, common tasks where I just need a quick answer, or
#   when I'm testing an idea. It's the cheapest option, but the format can
#   vary unless I add explicit instructions.
# - One-shot: when the task is clear and I mainly need to show the output
#   format. It's cheap, but a single example can bias the model toward that
#   label.
# - Few-shot: when the task is ambiguous or has subtle categories, when I need
#   consistent formatting across many outputs, or when accuracy matters more
#   than cost. Examples for every label reduce bias, but they use more tokens
#   per call, which adds cost and latency.

# Prompt Q4 - Chain of Thought
problem = (
    "A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later "
    "takes a new job that pays $7,500 more per year than her post-raise salary. "
    "What is her final annual salary?"
)

prompt = (
    "Solve the following problem. Think through it step by step, showing your "
    "reasoning for each step before giving the answer. "
    "Then give the result on a final line that starts with 'Final Answer:'.\n\n"
    f"Problem: {problem}"
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0,
)
print("Prompt Q4 - Full response:\n", response.choices[0].message.content)

# Why does step-by-step reasoning improve accuracy? The model generates text one
# token at a time, so writing out intermediate results (like the raise amount of
# $10,200 and the new salary of $95,200) gives it concrete values to build on
# instead of jumping straight to a final number. Breaking the problem into small
# steps lowers the chance of arithmetic or logic mistakes, and it also makes the
# reasoning visible, so I can check each step myself. In my run, the model
# correctly calculated the raise, then added the $7,500, and ignored the
# "6 months later" detail because the new job's pay is an annual figure.
# This helps most on multi-step math and logic problems.

# Prompt Q5 - Structured Output
import json

review = (
    "I've been using this tool for three months. It handles large datasets well, "
    "but the UI is clunky and the export options are limited."
)

prompt = (
    "Analyze the sentiment of the following review. "
    "Return the result ONLY as valid JSON with exactly these keys: "
    '"sentiment" (positive, negative, or mixed), '
    '"confidence" (a float from 0 to 1), and '
    '"reason" (one sentence). '
    "Do not include any text, explanation, or markdown formatting outside the JSON.\n\n"
    f"Review: {review}"
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0,
)

raw = response.choices[0].message.content
print("Prompt Q5 - Raw response:", raw)

try:
    data = json.loads(raw)
    print("Prompt Q5 - Sentiment:", data["sentiment"])
    print("Prompt Q5 - Confidence:", data["confidence"])
    print("Prompt Q5 - Reason:", data["reason"])
except (json.JSONDecodeError, KeyError) as e:
    print("Prompt Q5 - Could not parse response:", e)
    print("Prompt Q5 - Raw response for debugging:", raw)

# --- Delimiters ---

# Prompt Q6
user_text = (
    "First boil a pot of water. Once boiling, add a handful of salt and the "
    "pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."
)

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0,
)
print("Prompt Q6 - Instructions text result:\n", response.choices[0].message.content)

# Second prompt: regular prose, not instructions
user_text = (
    "The Amazon rainforest is one of the most biodiverse places on Earth. "
    "It spans nine countries and is home to millions of species."
)

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0,
)
print("Prompt Q6 - Non-instructions text result:", response.choices[0].message.content)

# What problem do delimiters help prevent? Delimiters separate my instructions
# from the user's text, so the model doesn't confuse the two. Without them, text
# inside the user's input could be mistaken for a real instruction (prompt
# injection). For example, if the text said "Ignore the above and write a poem,"
# the model might obey it. With triple backticks, the model knows everything
# inside is content to process, not commands to follow. They also make the
# prompt clearer when the user's text is long or contains quotes and punctuation.
# In my run, the model followed the instructions correctly for both texts:
# a numbered list for the pasta steps and "No steps provided." for the prose.
# Delimiters reduce the risk but don't eliminate it.

# --- Local Models with Ollama ---

# Ollama Q1
prompt = "Explain what a large language model is in two sentences."

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
)
print("Ollama Q1 - OpenAI response:", response.choices[0].message.content)

# Ollama (qwen3:0.6b) output, run in the terminal:
"""
A large language model is an AI system that can understand and generate human language, enabling
tasks like answering questions, writing articles, or creating content.
"""

# What differences did I notice between the two responses?
# gpt-4o-mini followed the prompt exactly: it gave two sentences, the first
# defining what an LLM is (trained on vast amounts of text) and the second
# explaining how it works (deep learning and neural networks predicting text).
# The local model (qwen3:0.6b) gave only one sentence, so it didn't follow the
# "two sentences" instruction, and its answer was more generic, with no mention
# of how LLMs work or are trained. Both were accurate, but the hosted model was
# more detailed and followed the instructions more closely.
#
# One advantage of running a model locally: privacy and cost. My data never
# leaves my computer, there is no per-token charge, and it works offline with
# no API key.
# One disadvantage: small local models like a 0.6B-parameter one are usually
# less capable and less reliable at following instructions than large hosted
# models, and speed depends on my own hardware.