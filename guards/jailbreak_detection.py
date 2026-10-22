import re
from transformers import pipeline
from llama import LlamaGuard, THRESHOLD
from dotenv import load_dotenv

load_dotenv()

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


def document_sanitizer(
    documents: list[dict],
    llama_guard: LlamaGuard,
    threshold: float = THRESHOLD,
    batch_size: int = 32,
) -> list[dict]:
    """
    Check documents sentence by sentence using Llama Prompt Guard.

    Sentences classified as MALICIOUS with confidence >= threshold
    are removed. Other sentences are retained.

    Expected document format:
        [
            {"page_number": 1, "content": "..."},
            {"page_number": 2, "content": "..."},
        ]

    Returns:
        A new list of documents with flagged sentences removed.
        The original documents are not modified.
    """
    sanitized_documents = []
    sentences_by_document = []
    all_sentences = []

    # Split each document into sentences.
    for document in documents:
        content = document.get("content", "")

        if not isinstance(content, str):
            raise TypeError("Document 'content' must be a string.")

        sentences = paragraph_to_sentences(content)

        sentences_by_document.append(sentences)
        all_sentences.extend(sentences)

    # Classify sentences in batches.
    results = []

    for start in range(0, len(all_sentences), batch_size):
        batch = all_sentences[start:start + batch_size]
        batch_results = llama_guard.check_batch(batch)
        results.extend(batch_results)

        # print(
        #     f"[Llama Guard] Checked "
        #     f"{min(start + len(batch), len(all_sentences))}"
        #     f"/{len(all_sentences)} sentences"
        # )

    if len(results) != len(all_sentences):
        raise ValueError(
            "The number of Llama Guard results does not match "
            "the number of sentences."
        )

    result_index = 0

    for document, sentences in zip(documents, sentences_by_document):
        clean_sentences = []

        for sentence in sentences:
            result = results[result_index]
            result_index += 1

            label = result["label"]
            confidence = result["confidence"]

            is_jailbreak = (
                label == "MALICIOUS"
                and confidence >= threshold
            )

            if is_jailbreak:
                print(
                    f"[Llama Guard Document Check] REMOVED | "
                    f"label={label} | "
                    f"confidence={confidence:.4f} | "
                    f"sentence={sentence!r}"
                )
                continue

            # print(
            #     f"[Llama Guard] KEPT | "
            #     f"label={label} | "
            #     f"confidence={confidence:.4f} | "
            #     f"sentence={sentence!r}"
            # )

            clean_sentences.append(sentence)

        sanitized_document = document.copy()
        sanitized_document["content"] = " ".join(clean_sentences)

        sanitized_documents.append(sanitized_document)

    return sanitized_documents


def input_check(
        query: str, 
        llama_guard: LlamaGuard, 
        threshold: float = THRESHOLD
    ) -> bool:
    """
    Validate a user prompt using Llama Prompt Guard.

    Returns:
        True  -> Prompt is considered safe.
        False -> Prompt is malicious, insufficiently confident,
                 empty, invalid, or could not be checked.
    """
    if not isinstance(query, str) or not query.strip():
        return False

    try:
        result = llama_guard.check(query.strip())

        label = result["label"]
        confidence = result["confidence"]

        is_safe = (
            label == "BENIGN"
            and confidence >= threshold
        )

        print(
            f"[Llama Guard Input Check] "
            f"label={label}, confidence={confidence:.4f}, "
            f"accepted={is_safe}"
        )

        return is_safe

    except Exception as exc:
        print(f"[Llama Guard Input Check] Verification failed: {exc}")
        return False

