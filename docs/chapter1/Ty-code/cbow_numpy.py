"""Train a CBOW model with full softmax and hand-written NumPy gradients."""

import re
from pathlib import Path
from typing import NamedTuple

import numpy as np


def preprocess_corpus(text):
    """Lowercase and tokenize each line independently."""
    sentences = []
    for line in text.splitlines():
        tokens = re.findall(r"[a-z]+", line.lower())
        if tokens:
            sentences.append(tokens)
    return sentences


def build_vocab(sentences):
    """Assign IDs in order of first occurrence."""
    id_to_word = list(dict.fromkeys(word for sentence in sentences for word in sentence))
    word_to_id = {word: word_id for word_id, word in enumerate(id_to_word)}
    return word_to_id, id_to_word


def generate_training_data(sentences, word_to_id, window_size=2):
    """Return (context_ids, target_id) pairs, keeping sentence boundaries."""
    if window_size < 1:
        raise ValueError("window_size must be at least 1")

    samples = []
    for sentence in sentences:
        word_ids = [word_to_id[word] for word in sentence]
        for center, target_id in enumerate(word_ids):
            left = max(0, center - window_size)
            right = min(len(word_ids), center + window_size + 1)
            context_ids = word_ids[left:center] + word_ids[center + 1:right]
            if context_ids:
                samples.append((context_ids, target_id))
    return samples


def initialize_parameters(vocab_size, embedding_dim, rng):
    """Initialize both weights randomly and start the output bias at zero."""
    scale = 1.0 / np.sqrt(embedding_dim)
    W_in = rng.normal(0.0, scale, size=(vocab_size, embedding_dim))  # (V, d)
    W_out = rng.normal(0.0, scale, size=(embedding_dim, vocab_size))  # (d, V)
    b = np.zeros(vocab_size)  # (V,)
    return W_in, W_out, b


class ForwardPass(NamedTuple):
    context_vectors: np.ndarray  # (C, d)
    h: np.ndarray  # (d,)
    logits: np.ndarray  # (V,)
    log_probs: np.ndarray  # (V,)
    probs: np.ndarray  # (V,)


def compute_loss(log_probs, target_id):
    """Cross-entropy with a one-hot target: L = -log(p[target_id])."""
    return float(-log_probs[target_id])


class CBOW:
    def __init__(self, word_to_id, id_to_word, embedding_dim, rng):
        self.word_to_id = word_to_id
        self.id_to_word = id_to_word
        self.W_in, self.W_out, self.b = initialize_parameters(
            len(id_to_word), embedding_dim, rng
        )

    def forward(self, context_ids):
        if len(context_ids) == 0:
            raise ValueError("At least one context word is required")

        # Look up the C rows that a (C, V) one-hot matrix would select.
        context_vectors = self.W_in[context_ids, :]  # (C, d)
        h = context_vectors.mean(axis=0)  # (d,)
        logits = h @ self.W_out + self.b  # (V,)

        # Shift by max(z) before exponentiating. Compute log(p) directly
        # so cross-entropy stays finite even if a probability underflows to 0.
        shifted = logits - logits.max()  # (V,)
        log_probs = shifted - np.log(np.exp(shifted).sum())  # (V,)
        probs = np.exp(log_probs)  # (V,)
        return ForwardPass(context_vectors, h, logits, log_probs, probs)

    def backward(self, context_ids, target_id, result):
        # Softmax + cross-entropy: dL/dz = p - y, with one-hot y.
        dz = result.probs.copy()  # (V,)
        dz[target_id] -= 1.0

        dW_out = np.outer(result.h, dz)  # (d, V): h_i * dz_j
        db = dz  # (V,): z includes b with coefficient 1
        dh = dz @ self.W_out.T  # (d,): sum over the V output classes

        # Mean pooling gives dh/C to each context occurrence.
        # Add each occurrence to its row, including repeat visits to a row.
        dW_in = np.zeros_like(self.W_in)  # (V, d)
        per_context_gradient = dh / len(context_ids)  # (d,)
        for word_id in context_ids:
            dW_in[word_id] += per_context_gradient

        return dW_in, dW_out, db

    def predict(self, context_words, top_k=5):
        """Return (word, probability) pairs in descending probability order."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        try:
            context_ids = [self.word_to_id[word.lower()] for word in context_words]
        except KeyError as error:
            raise ValueError(f"Unknown context word: {error.args[0]!r}") from error

        probs = self.forward(context_ids).probs
        ranked_ids = np.argsort(-probs, kind="stable")[:top_k]
        return [(self.id_to_word[word_id], float(probs[word_id])) for word_id in ranked_ids]


def mean_loss(model, samples):
    """Evaluate all samples with one fixed set of parameters."""
    return float(np.mean([
        compute_loss(model.forward(context_ids).log_probs, target_id)
        for context_ids, target_id in samples
    ]))


def train(model, samples, rng, learning_rate=0.1, epochs=150):
    """Run shuffled SGD and record full-corpus mean loss at each epoch end."""
    if not samples:
        raise ValueError("Training requires at least one sample with context")

    loss_history = [mean_loss(model, samples)]  # Index 0 is before any updates.
    print(f"Initial mean loss: {loss_history[0]:.6f}")
    for epoch in range(1, epochs + 1):
        for sample_id in rng.permutation(len(samples)):
            context_ids, target_id = samples[sample_id]
            result = model.forward(context_ids)
            dW_in, dW_out, db = model.backward(context_ids, target_id, result)

            # Compute all gradients at the old parameters before updating them.
            model.W_in -= learning_rate * dW_in
            model.W_out -= learning_rate * dW_out
            model.b -= learning_rate * db

        loss_history.append(mean_loss(model, samples))
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            print(f"Epoch {epoch:3d}/{epochs}: mean loss = {loss_history[-1]:.6f}")

    print(f"Final mean loss:   {loss_history[-1]:.6f}")
    return loss_history


def cosine_similarity(vector_a, vector_b):
    """Compute a·b / (||a|| ||b||) for two nonzero embedding vectors."""
    norm_product = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if norm_product == 0:
        raise ValueError("Cosine similarity requires two nonzero vectors")
    return float(np.dot(vector_a, vector_b) / norm_product)


def gradient_check(model, context_ids, target_id, epsilon=1e-5):
    """Check selected entries of all three gradients using central differences."""
    result = model.forward(context_ids)
    dW_in, dW_out, db = model.backward(context_ids, target_id, result)
    other_id = context_ids[-1]
    checks = (
        ("W_in", model.W_in, dW_in, [(context_ids[0], 0), (other_id, -1), (target_id, 0)]),
        ("W_out", model.W_out, dW_out, [(0, target_id), (-1, other_id)]),
        ("b", model.b, db, [(target_id,), (other_id,)]),
    )
    print("\nFinite-difference gradient check")
    max_error = 0.0
    for name, parameter, analytical, indices in checks:
        for index in indices:
            original = parameter[index]
            try:
                parameter[index] = original + epsilon
                loss_plus = compute_loss(model.forward(context_ids).log_probs, target_id)
                parameter[index] = original - epsilon
                loss_minus = compute_loss(model.forward(context_ids).log_probs, target_id)
            finally:
                parameter[index] = original

            numerical = (loss_plus - loss_minus) / (2 * epsilon)
            manual = analytical[index]
            error = abs(numerical - manual)
            max_error = max(max_error, error)
            print(f"  {name}{index!s:>8}  manual={manual:+.8f}  "
                  f"numeric={numerical:+.8f}  abs_error={error:.2e}")
            if not np.isclose(numerical, manual, rtol=1e-5, atol=1e-7):
                raise AssertionError(f"Gradient check failed for {name}{index}")
    print(f"Gradient check passed (max absolute error {max_error:.2e})")
    return max_error


def show_example(model, context_ids, target_id):
    """Trace one sample before training, including the one-hot equivalence."""
    result = model.forward(context_ids)
    dW_in, dW_out, db = model.backward(context_ids, target_id, result)
    context_words = [model.id_to_word[word_id] for word_id in context_ids]
    target_word = model.id_to_word[target_id]

    one_hot = np.zeros((len(context_ids), len(model.id_to_word)))  # (C, V)
    one_hot[np.arange(len(context_ids)), context_ids] = 1.0
    one_hot_vectors = one_hot @ model.W_in  # (C, d)

    print(f"\nExample: {context_words} -> {target_word!r} before training")
    print(f"  Context IDs = {context_ids}, target ID = {target_id}")
    print(f"  W_in {model.W_in.shape}, W_out {model.W_out.shape}, b {model.b.shape}")
    print(f"  Context one-hot matrix: {one_hot.shape}")
    print(f"  Context vectors:        {result.context_vectors.shape}")
    print(f"  One-hot multiplication equals lookup: "
          f"{np.array_equal(one_hot_vectors, result.context_vectors)}")
    print(f"  Mean h:                 {result.h.shape}, first four = {result.h[:4]}")
    print(f"  Logits z:               {result.logits.shape}, z[{target_word}] = "
          f"{result.logits[target_id]:.6f}")
    print(f"  Probabilities p:        {result.probs.shape}, sum = {result.probs.sum():.6f}")
    print(f"  p[{target_word}] = {result.probs[target_id]:.6f}, "
          f"loss = {compute_loss(result.log_probs, target_id):.6f}")
    print(f"  dW_in {dW_in.shape}, dW_out {dW_out.shape}, db {db.shape}")


def main():
    corpus_path = Path(__file__).with_name("corpus.txt")
    sentences = preprocess_corpus(corpus_path.read_text(encoding="utf-8"))
    word_to_id, id_to_word = build_vocab(sentences)
    window_size = 2
    embedding_dim = 100
    rng = np.random.default_rng(42)

    samples = generate_training_data(sentences, word_to_id, window_size)
    model = CBOW(word_to_id, id_to_word, embedding_dim, rng)
    print(f"Sentences: {len(sentences)}, vocabulary: {len(id_to_word)}, "
          f"samples: {len(samples)}, window: {window_size}, dimension: {embedding_dim}")

    context_words = ["the", "quick", "fox", "jumps"]
    target_word = "brown"
    context_ids = [word_to_id[word] for word in context_words]
    target_id = word_to_id[target_word]
    show_example(model, context_ids, target_id)
    gradient_check(model, context_ids, target_id)

    print("\nTraining")
    train(model, samples, rng, learning_rate=0.1, epochs=150)

    predictions = model.predict(context_words, top_k=len(id_to_word))
    print(f"\nTop 5 predictions for {context_words}")
    for rank, (word, probability) in enumerate(predictions[:5], start=1):
        print(f"  {rank}. {word:<8} {probability:.6f}")

    target_rank, target_probability = next(
        (rank, probability)
        for rank, (word, probability) in enumerate(predictions, start=1)
        if word == target_word
    )
    print(f"  Target {target_word!r}: rank {target_rank}/{len(id_to_word)}, "
          f"probability {target_probability:.6f}")

    target_vector = model.W_in[target_id]
    similarities = [
        (word, cosine_similarity(target_vector, model.W_in[word_id]))
        for word_id, word in enumerate(id_to_word)
        if word_id != target_id
    ]
    similarities.sort(key=lambda pair: pair[1], reverse=True)
    print(f"\nTop 5 cosine similarities to {target_word!r} (input embeddings)")
    for word, similarity in similarities[:5]:
        print(f"  {word:<8} {similarity:+.6f}")
    print("These scores illustrate training on a small corpus. "
          "They provide limited evidence about linguistic similarity.")


if __name__ == "__main__":
    main()
