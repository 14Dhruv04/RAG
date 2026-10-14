import re
from transformers import pipeline
from guardrails import Guard
from guardrails_ai.detect_jailbreak import DetectJailbreak
from dotenv import load_dotenv
# import sys
# sys.path.append("/home/dhruv/Desktop/personal/RAG")
# # from parser import PDFparser

load_dotenv()

THRESHOLD = 0.8

detector = DetectJailbreak(
        threshold = THRESHOLD,
        on_fail = "noop",
        use_local = True,
        device = "cpu"
    )

def paragraph_to_sentences(paragraph: str) -> list[str]:
    # Remove null characters and replace newlines/tabs with spaces
    paragraph = paragraph.replace('\0', '')
    paragraph = re.sub(r'[\n\r\t]+', ' ', paragraph)

    # Normalize multiple spaces
    paragraph = re.sub(r'\s+', ' ', paragraph).strip()

    # Split into sentences using punctuation
    sentences = re.split(r'(?<=[.!?])\s+', paragraph)

    # Remove empty sentences
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def document_sanitizer(documents: list[dict]) -> list[dict]:
    sanitized_documents = []
    sentences_by_document = []
    all_sentences = []

    for document in documents:
        content = document.get("content", "")

        if not isinstance(content, str):
            raise TypeError("Document 'content' must be a string.")

        sentences = paragraph_to_sentences(content)
        sentences_by_document.append(sentences)
        all_sentences.extend(sentences)

    scores = []
    batch_size = 32
    for start in range(0, len(all_sentences), batch_size):
        batch = all_sentences[start:start + batch_size]
        batch_scores = detector.predict_jailbreak(batch)
        scores.extend(batch_scores)

    if len(scores) != len(all_sentences):
        raise ValueError(
            "The number of jailbreak scores does not match the number of sentences."
        )

    # Reconstruct each page using only sentences below the threshold.
    score_index = 0
    for document, sentences in zip(documents, sentences_by_document):
        clean_sentences = []

        for sentence in sentences:
            score = scores[score_index]
            score_index += 1

            if score > THRESHOLD:
                print("\nREMOVED: ", sentence)
                continue

            clean_sentences.append(sentence)

        sanitized_document = document.copy()
        sanitized_document["content"] = " ".join(clean_sentences)

        sanitized_documents.append(sanitized_document)

    return sanitized_documents

def input_check(query) -> bool:
    guard = Guard().use(detector)
    return guard.validate(query).validation_passed
