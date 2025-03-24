from evaluate import load

# Load metrics
bleu = load("bleu")
rouge = load("rouge")
bertscore = load("bertscore")

# Example hypothesis & reference texts
predictions = [
    "Paris is the capital of France.",
    "The Earth revolves around the Sun."
]
references = [
    ["The capital city of France is Paris.", "Paris is the capital of France."],
    ["The Sun is at the center of the Solar System.", "Earth orbits the Sun."]
]

# Compute BLEU (Bilingual Evaluation Understudy)
# Measures how similar the generated text is to the reference text.
# Higher is better (range: 0 to 100).
bleu_score = bleu.compute(predictions=predictions, references=references)
print("Printing BLEU Score: ")
print(bleu_score)

# Compute ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
# Measures similarity between model-generated and reference text using n-grams.
# Higher is better.
# Types:
    # ROUGE-1 (Unigrams): Measures how many single words match.
    # ROUGE-2 (Bigrams): Measures how many two-word sequences match.
    # ROUGE-L (Longest Common Subsequence): Captures overall fluency.
rouge_score = rouge.compute(predictions=predictions, references=references)
print("Printing ROUGE Score: ")
print(rouge_score)

# Compute BERTScore (semantic similarity)
bert_score = bertscore.compute(predictions=predictions, references=references, lang="en")
print("Printing BERT Score: ")
print(bert_score)