import tensorflow as tf
import transformers
import numpy as np
from huggingface_hub import from_pretrained_keras
import pandas as pd
import nltk


# Ensure NLTK data is downloaded
def setup_nltk():
    """Download required NLTK data packages."""
    try:
        required_packages = ['punkt', 'stopwords', 'punkt_tab']
        for package in required_packages:
            try:
                nltk.data.find(f'tokenizers/{package}')
            except LookupError:
                print(f"Downloading {package}...")
                nltk.download(package, quiet=True)
        return True
    except Exception as e:
        print(f"Error setting up NLTK: {str(e)}")
        return False


class BertSemanticDataGenerator(tf.keras.utils.Sequence):
    """Generates batches of data for BERT processing."""

    def __init__(
            self,
            sentence_pairs,
            labels=None,
            batch_size=32,
            shuffle=True,
            include_targets=True,
    ):
        self.sentence_pairs = sentence_pairs
        self.labels = labels
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.include_targets = include_targets
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )
        self.indexes = np.arange(len(self.sentence_pairs))
        self.on_epoch_end()

    def __len__(self):
        return len(self.sentence_pairs) // self.batch_size

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size: (idx + 1) * self.batch_size]
        sentence_pairs = self.sentence_pairs[indexes]

        encoded = self.tokenizer.batch_encode_plus(
            sentence_pairs.tolist(),
            add_special_tokens=True,
            max_length=128,
            return_attention_mask=True,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_tensors="tf",
        )

        input_ids = np.array(encoded["input_ids"], dtype="int32")
        attention_masks = np.array(encoded["attention_mask"], dtype="int32")
        token_type_ids = np.array(encoded["token_type_ids"], dtype="int32")

        if self.include_targets:
            labels = np.array(self.labels[indexes], dtype="int32")
            return [input_ids, attention_masks, token_type_ids], labels
        else:
            return [input_ids, attention_masks, token_type_ids]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


def simple_tokenize(text):
    """Simple word tokenization without relying on NLTK's punkt."""
    # Basic cleaning
    text = text.lower().strip()
    # Split on whitespace and punctuation
    words = text.split()
    return words


def calculate_similarity_score(teacher_answer, student_answer, model):
    """Calculate similarity between answers using BERT."""
    sentence_pairs = np.array([[str(teacher_answer), str(student_answer)]])
    test_data = BertSemanticDataGenerator(
        sentence_pairs, labels=None, batch_size=1, shuffle=False, include_targets=False,
    )

    try:
        probs = model.predict(test_data[0], verbose=0)[0]
        labels = ["Contradiction", "Perfect", "Neutral"]
        return {labels[i]: float(probs[i]) for i, _ in enumerate(labels)}
    except Exception as e:
        raise Exception(f"Error in similarity calculation: {str(e)}")


def calculate_length_score(teacher_answer, student_answer):
    """Calculate length-based score without NLTK dependency."""
    teacher_words = set(simple_tokenize(teacher_answer))
    student_words = set(simple_tokenize(student_answer))

    teacher_length = len(teacher_words)
    student_length = len(student_words)

    if teacher_length == 0:
        return 1.0

    ratio = student_length / teacher_length

    if ratio < 0.5:  # Too short
        return 0.5
    elif ratio > 1.5:  # Too long
        return 0.8
    return 1.0


def process_answers(teacher_file, student_file):
    """Process answers from Excel files and calculate scores."""
    try:
        # Load model
        print("Loading BERT model...")
        model = from_pretrained_keras("keras-io/bert-semantic-similarity")

        # Read Excel files
        print("Reading Excel files...")
        teacher_df = pd.read_excel(teacher_file)
        student_df = pd.read_excel(student_file)

        teacher_answer = str(teacher_df.iloc[0, 0])
        student_answer = str(student_df.iloc[0, 0])

        # Calculate scores
        print("Calculating scores...")
        similarity_scores = calculate_similarity_score(teacher_answer, student_answer, model)
        length_score = calculate_length_score(teacher_answer, student_answer)

        # Calculate final score
        perfect_score = similarity_scores["Perfect"]
        final_score = int(perfect_score * length_score * 100)

        return {
            "teacher_answer": teacher_answer,
            "student_answer": student_answer,
            "similarity_scores": {
                "contradiction": int(similarity_scores["Contradiction"] * 100),
                "perfect": int(similarity_scores["Perfect"] * 100),
                "neutral": int(similarity_scores["Neutral"] * 100)
            },
            "length_score": length_score,
            "final_score": final_score
        }

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    print("Starting answer evaluation...")

    # Process files
    results = process_answers("teacher_answers.xlsx", "student_answers_offtopic.xlsx")

    if "error" in results:
        print(f"\nError: {results['error']}")
    else:
        print("\nAnswer Evaluation Results:")
        print("-" * 50)
        print(f"Teacher's Answer: {results['teacher_answer']}")
        print(f"Student's Answer: {results['student_answer']}")
        print("\nScores:")
        print(f"Perfect Match: {results['similarity_scores']['perfect']}%")
        print(f"Neutral: {results['similarity_scores']['neutral']}%")
        print(f"Contradiction: {results['similarity_scores']['contradiction']}%")
        print(f"Length Score: {results['length_score']:.2f}")
        print(f"\nFinal Score: {results['final_score']}/100")