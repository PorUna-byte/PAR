compare_prompt_for_chat = """Please act as an impartial evaluator to assess the quality of two responses from different AI assistants to an incomplete dialogue between a user (<|user|>) and an AI assistant (<|assistant|>). The dialogue will be missing the last turn, and both Assistant-A (<Assistant-A response>) and Assistant-B (<Assistant-B response>) are expected to complete it. Focus your evaluation on the following five aspects:
1. Clarity and Relevance: Responses should be concise, directly addressing the question. They should use clear, natural language and remain on-topic.
2. Accuracy and Honesty: Responses must provide factual, truthful information. Disclose limitations or uncertainties when necessary.
3. Ethics and Appropriateness: Ensure the responses are free from harmful, offensive, or discriminatory content.
4. Engagement and Depth: Responses should be engaging, educational, and sufficiently detailed to comprehensively address the user's question.
5. Structure and Creativity: Responses should be logically organized and show originality or adaptability when necessary.

Note: The quality of the responses should not be judged solely by their length. Both brevity and detail are important depending on the context of the question.
You will be given an incomplete dialogue (<question>) with the last turn left blank. Assistant-A (<Assistant-A response>) and Assistant-B (<Assistant-B response>) have each provided a response to complete the dialogue. Your task is to evaluate each response based on the five criteria above and provide a comparison.

Evaluation Format:
Assistant-A's Response:
(Evaluate the quality of Assistant-A's response based on the five aspects mentioned above.)
Assistant-B's Response:
(Evaluate the quality of Assistant-B's response based on the five aspects mentioned above.)
Comparison and Analysis:
Compare and contrast the responses from Assistant-A and Assistant-B to determine which one is more effective overall. Justify your reasoning clearly and concisely.

At the end, output the comparison result for both responses in the following format:
Better: X (X is A, B, or N, representing A is better, B is better, or both are of equal quality)
"""



compare_prompt_for_summary = """You are a content evaluator. Your task is to assess and compare two summaries of a given post based on the following criteria: **coverage of key points**, **conciseness and precision**, **accuracy and faithfulness**, and **coherence and fluency**.

### **Scoring Criteria:**
1. **Coverage of Key Points:**  
   Does the summary capture the main ideas and key information from the original post without omitting critical details? Does it accurately reflect the post's scope?

2. **Conciseness and Precision:**  
   Is the summary concise and to the point, avoiding unnecessary verbosity while ensuring all essential points are included? Does it strike a balance between brevity and completeness?

3. **Accuracy and Faithfulness:**  
   Does the summary faithfully represent the content and meaning of the original post? Are there any factual errors, misinterpretations, or deviations from the original information?

4. **Coherence and Fluency:**  
   Is the summary well-structured, logically flowing from one point to the next? Is it grammatically sound, easy to read, and free of awkward phrasing?

The most important principle is to never judge a summary solely by its length; you should not give any preference based on how long or short the summary is.
You will be provided with a title and post (<title&post>), along with two summaries: one from Assistant-A (<Assistant-A summary>) and another from Assistant-B (<Assistant-B summary>).

Please evaluate each summary according to the above criteria and provide an objective analysis. After reviewing the summaries, compare them and determine which one is better.

### **Evaluation Format:**
- **Assistant-A's Summary:**
  (Evaluate Assistant-A's summary based on the four criteria)

- **Assistant-B's Summary:**
  (Evaluate Assistant-B's summary based on the four criteria)

- **Comparison and Analysis:**
  Compare the summaries of Assistant-A and Assistant-B and determine which is of higher quality.

### **Final Output Format:**
- Better: X (Where X is A, B, or N, indicating whether Assistant-A is better, Assistant-B is better, or if both summaries are of equal quality.)
"""

summary_prompt_template = """Please summary the following content:\n\nTitle:{title}\nPost:{post}\n\nThe summary is:"""