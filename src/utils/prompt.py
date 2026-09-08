compare_prompt_for_chat = """Please act as an impartial evaluator to assess the quality of two responses from different AI assistants to an incomplete dialogue between a user (<|user|>) and an AI assistant (<|assistant|>). The dialogue will be missing the last turn, and both Assistant-A (<Assistant-A response>) and Assistant-B (<Assistant-B response>) are expected to complete it. Focus your evaluation on the following five aspects:
1. Clarity and Relevance: Responses should be concise, directly addressing the question. They should use clear, natural language and remain on-topic.
2. Accuracy and Honesty: Responses must provide factual, truthful information. Disclose limitations or uncertainties when necessary.
3. Ethics and Appropriateness: Ensure the responses are free from harmful, offensive, or discriminatory content.
4. Engagement and Depth: Responses should be engaging, educational, and sufficiently detailed to comprehensively address the user question.
5. Structure and Creativity: Responses should be logically organized and show originality or adaptability when necessary.

Note: The quality of the responses should not be judged solely by their length. Both brevity and detail are important depending on the context of the question.
You will be given an incomplete dialogue (<question>) with the last turn left blank. Assistant-A (<Assistant-A response>) and Assistant-B (<Assistant-B response>) have each provided a response to complete the dialogue. Your task is to evaluate each response based on the five criteria above and provide a comparison.
"""