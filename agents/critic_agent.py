def critic_agent(question, chunks, embed_model, answer, threshold=0.25):
    if not answer or len(answer.strip()) < 5:
        return "Not found in document."
    return answer
