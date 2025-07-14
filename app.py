import sys
import os
import torch
import numpy as np
import pandas as pd
import random
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import DataFrameLoader
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage, AIMessage

# =========================
# Environment & Model Setup
# =========================

def setup_environment():
    load_dotenv()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in .env file")
    chat_model = ChatGroq(model="mistral-saba-24b", api_key=groq_api_key)
    return device, embedding_model, chat_model

# =========================
# Data Loading
# =========================

def load_data():
    df = pd.read_csv('Software Questions.csv', encoding='ISO-8859-1')
    loader = DataFrameLoader(df, page_content_column="Question")
    questions = loader.load()
    questions_text = [doc.page_content for doc in questions]
    answers_text = [doc.metadata['Answer'] for doc in questions]
    categories = list(set(doc.metadata["Category"] for doc in questions))
    questions_by_category = {category: [] for category in categories}
    for doc in questions:
        category = doc.metadata["Category"]
        questions_by_category[category].append(doc)
    return questions, questions_text, answers_text, categories, questions_by_category



def load_embeddings():
    question_embeddings = np.load('question_embeddings.npy')
    answer_embeddings = np.load('answer_embeddings.npy')
    category_embeddings = np.load('category_embeddings.npy')
    return question_embeddings, answer_embeddings, category_embeddings

# =========================
# Utility Functions
# =========================

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def get_user_category(categories):
    print("Bot: Please choose a category from the following options:")
    for idx, category in enumerate(categories, start=1):
        print(f"{idx}. {category}")
    while True:
        choice = input("You: ")
        if choice.isdigit() and 1 <= int(choice) <= len(categories):
            return categories[int(choice) - 1]
        else:
            print("Bot: Invalid choice. Please enter a number corresponding to a category.")

def generate_response(chat_model, user_input, correct_answer):
    messages = [
        SystemMessage(content="You are a helpful interview bot that evaluates user responses and gives them a score out of 5."),
        HumanMessage(content=f"User's answer: {user_input}\nExpected answer: {correct_answer}\nEvaluate correctness and give a response.")
    ]
    response = chat_model.invoke(messages)
    return response.content

def get_new_question(previous_question_idx, category_questions, asked_questions, question_embeddings):
    while True:
        new_idx = random.randint(0, len(category_questions) - 1)
        if new_idx == previous_question_idx or new_idx in asked_questions:
            continue
        if previous_question_idx is not None:
            similarity = cosine_similarity(
                question_embeddings[previous_question_idx],
                question_embeddings[new_idx]
            )
            if similarity > 0.7:
                continue
        asked_questions.add(new_idx)
        return new_idx

# =========================
# Main Interactive Loop
# =========================

def interactive_qa(
    questions_text, answers_text, categories, questions_by_category,
    question_embeddings, answer_embeddings, embedding_model, chat_model
):
    chat_history = []
    asked_questions = set()
    print("Bot: Hi! Let's start the conversation.")
    sys.stdout.flush()

    chosen_category = get_user_category(categories)
    category_questions = questions_by_category[chosen_category]

    current_question_idx = get_new_question(None, category_questions, asked_questions, question_embeddings)
    current_question = category_questions[current_question_idx].page_content
    correct_answer = category_questions[current_question_idx].metadata['Answer']

    print(f"Bot: {current_question}")
    sys.stdout.flush()
    chat_history.append(AIMessage(content=current_question))

    while True:
        user_response = input("You: ")
        if user_response.lower() in ["exit", "quit", "stop"]:
            print("Bot: Thanks for the chat! Here's your conversation history:")
            for entry in chat_history:
                print(entry)
            break

        chat_history.append(HumanMessage(content=user_response))
        user_response_embedding = embedding_model.encode(user_response)
        expected_answer_embedding = answer_embeddings[current_question_idx]
        similarity = cosine_similarity(user_response_embedding, expected_answer_embedding)

        feedback = generate_response(chat_model, user_response, correct_answer)
        print(f"Bot: {feedback}")
        sys.stdout.flush()
        chat_history.append(AIMessage(content=feedback))

        new_question_idx = get_new_question(current_question_idx, category_questions, asked_questions, question_embeddings)
        new_question = category_questions[new_question_idx].page_content
        correct_answer = category_questions[new_question_idx].metadata['Answer']
        print(f"Bot: {new_question}")
        current_question_idx = new_question_idx
        sys.stdout.flush()
        chat_history.append(AIMessage(content=new_question))

# =========================
# Main Entrypoint
# =========================

def main():
    device, embedding_model, chat_model = setup_environment()
    questions, questions_text, answers_text, categories, questions_by_category = load_data()
    question_embeddings, answer_embeddings, category_embeddings = load_embeddings()
    interactive_qa(
        questions_text, answers_text, categories, questions_by_category,
        question_embeddings, answer_embeddings, embedding_model, chat_model
    )

if __name__ == "__main__":
    main()
