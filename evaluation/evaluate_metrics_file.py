from evaluate import load
import sys

from evaluation.evaluate_metrics import main as evaluation_updated


def read_results_from_file(filename):
    with open(filename) as file:
        content = file.readlines()
    
    references = list()
    predictions = list()
    for line in content:
        if "Best Answer: " in line:
            references.append(line.replace("Best Answer: ", ""))
        if "Model Response: " in line:
            predictions.append(line.replace("Model Response: ", ""))
    return references, predictions

def convert_dict_to_string(dictionary: dict):
    string = ""
    for key, value in dictionary.items():
        if (type(value) == dict):
            string += "{}: {}\n".format(key, convert_dict_to_string(value))
        else:
            string += "{}: {}\n".format(key, value)
    return string

def evaluate_inference(filepath):
    # Load metrics
    bleu = load("bleu")
    rouge = load("rouge")
    bertscore = load("bertscore")

    # Example hypothesis & reference texts
    # predictions = [
    #     "Paris is the capital of France.",
    #     "The Earth revolves around the Sun."
    # ]
    # references = [
    #     ["The capital city of France is Paris.", "Paris is the capital of France."],
    #     ["The Sun is at the center of the Solar System.", "Earth orbits the Sun."]
    # ]

    references, predictions = read_results_from_file(filepath)

    output_str = ""

    # Compute BLEU (Bilingual Evaluation Understudy)
    # Measures how similar the generated text is to the reference text.
    # Higher is better (range: 0 to 100).
    bleu_score = bleu.compute(predictions=predictions, references=references)
    output_str += "Printing BLEU Score: \n"
    output_str += convert_dict_to_string(bleu_score)
    output_str += "\n"

    # Compute ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
    # Measures similarity between model-generated and reference text using n-grams.
    # Higher is better.
    # Types:
        # ROUGE-1 (Unigrams): Measures how many single words match.
        # ROUGE-2 (Bigrams): Measures how many two-word sequences match.
        # ROUGE-L (Longest Common Subsequence): Captures overall fluency.
    rouge_score = rouge.compute(predictions=predictions, references=references)
    output_str += "Printing ROUGE Score: \n"
    output_str += convert_dict_to_string(rouge_score)
    output_str += "\n"

    # Compute BERTScore (semantic similarity)
    bert_score = bertscore.compute(predictions=predictions, references=references, lang="en")
    output_str += "Printing BERT Score: \n"
    output_str += convert_dict_to_string(bert_score)
    output_str += "\n"

    output_filename = filepath.replace("dataset_run", "evaluation")
    output_file = open(output_filename, "w")
    output_file.write(output_str)
    output_file.flush()
    output_file.close()

def evaluate_inference_updated(inference_results_filepath):
    evaluation_filename = inference_results_filepath.replace("dataset_run", "evaluation").replace(".txt", ".json")
    dataset_name = "truthful_qa"
    evaluation_updated(dataset_name, inference_results_filepath, evaluation_filename, from_root=True)


if (__name__ == "__main__"):
    evaluate_inference_updated(sys.argv[1])