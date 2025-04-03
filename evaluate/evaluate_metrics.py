from evaluate import load
import statistics
from datasets import load_from_disk
import json
import nltk
from nltk.translate.bleu_score import corpus_bleu

# Second Method
def calculate_bleu(predictions, references):
    # Tokenize sentences
    tokenized_predictions = [nltk.word_tokenize(prediction.lower()) for prediction in predictions]
    tokenized_references = [[nltk.word_tokenize(ref.lower()) for ref in refs] for refs in references]

    bleu_score = corpus_bleu(tokenized_references, tokenized_predictions)
    return bleu_score


def calculate_metrics(predictions, references):
    result = {}

    # Load metrics
    bleu = load("bleu")
    rouge = load("rouge")
    bertscore = load("bertscore")
    meteor = load("meteor")

    # METEOR (More Semantic-Based)
    # Accounts for synonyms, stemming, and paraphrasing (better than BLEU).
    # Works well for smaller datasets where exact n-gram matches are less frequent.
    meteor_score = meteor.compute(predictions=predictions, references=references)
    result["meteor"] = meteor_score["meteor"]
    

    # Compute BLEU (Bilingual Evaluation Understudy)
    # Measures how similar the generated text is to the reference text.
    # Higher is better (range: 0 to 100).
    bleu_score = bleu.compute(predictions=predictions, references=references)
    result["bleu"] = bleu_score

    # Second method BLEU
    bleu = calculate_bleu(predictions, references)
    result["corpus_bleu"] = bleu


    # Compute ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
    # Measures similarity between model-generated and reference text using n-grams.
    # Higher is better.
    # Types:
        # ROUGE-1 (Unigrams): Measures how many single words match.
        # ROUGE-2 (Bigrams): Measures how many two-word sequences match.
        # ROUGE-L (Longest Common Subsequence): Captures overall fluency.
    rouge_score = rouge.compute(predictions=predictions, references=references)
    result["rouge"] = rouge_score


    # # Compute BERTScore (semantic similarity)
    bert_score = bertscore.compute(predictions=predictions, references=references, lang="en")
    modified_bert = {
        "precision": statistics.mean(bert_score["precision"]),
        "recall": statistics.mean(bert_score["recall"]),
        "f1": statistics.mean(bert_score["f1"])
    }
    result["bert"] = modified_bert

    return result

def contains(pred, answers):
    """Returns True if any of the ground-truth answers appears in the predicted answer."""
    pred = pred.strip().lower()
    return any(ans.strip().lower() in pred for ans in answers)

dataset_name = "wmt16-de-en"

files_path = f"../results/{dataset_name}/reference"
file_name = "opt_zero_shot"
input_path = f"{files_path}/{file_name}.txt"
output_path = f"{files_path}/{file_name}.json"

predictions = []
# If a response has multiple lines because of hullicinations, take only first line.
# Remove the "." at the end of the response
with open(input_path, "r") as f:
    while line := f.readline():
        line = line.strip()
        if line.startswith("Model Response:"):
            model_response = line.split("Model Response:")[1].strip()
            predictions.append(model_response.split(".")[0])


dataset = load_from_disk(f"../dataset/{dataset_name}")

if (dataset_name == "trurthul_qa"):
    references = dataset["validation"]["correct_answers"][:len(predictions)]
    # Remove empty references
    filtered_references = [[item for item in row if item != ""] for row in references]

    result = calculate_metrics(predictions, filtered_references)

    with open(output_path, "w") as f:
        json.dump(result, f)

elif (dataset_name == "trivia_qa"):
    correct = 0
    total = len(predictions)
    for i in range(0, total):
        predicted_answer = predictions[i]
        ground_truths = dataset["validation"][i]["answer"]["normalized_aliases"]
        if contains(predicted_answer, ground_truths):
            correct += 1

    contains_score = correct / total
    json_dict = {
        "contains": contains_score
    }

    with open(output_path, "w") as f:
        json.dump(json_dict, f)
    print(f"Contains Score: {contains_score:.4f}")

elif (dataset_name == "wmt16-de-en"):
    references = [[x["en"]] for x in dataset["validation"]["translation"][:len(predictions)]]
    result = calculate_metrics(predictions, references)

    with open(output_path, "w") as f:
        json.dump(result, f)