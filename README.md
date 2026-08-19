# PrivRAG-Guard

Privacy-preserving middleware that sanitizes vector embeddings before they reach
a database. Defends RAG pipelines against embedding inversion attacks using
calibrated differential-privacy noise, semantic subspace projection, and keyed
orthogonal rotation — while preserving >98% cosine retrieval accuracy.

## Install

```bash
pip install -e ".[dev]"
```

## Library usage

```python
from privrag import PrivRAGGuard

guard = PrivRAGGuard(epsilon=1.0, noise_fraction=0.10, passphrase="shared-secret")
protected = guard.protect_batch(document_embeddings)
query = guard.protect_vector(query_embedding)

metrics = guard.benchmark_utility(document_embeddings, protected, k=5)
```

### Sharing keys across services

Derive deterministically from a passphrase (inject via secret manager):

```python
guard = PrivRAGGuard(passphrase=os.environ["PRIVRAG_KEY"])
```

Or persist a key file from a protected volume:

```python
guard.protect_vector(sample)  # initializes rotation
guard.save_key("/run/secrets/privrag.key")

other = PrivRAGGuard.from_key_file("/run/secrets/privrag.key")
```

### Adapters

```python
from privrag.adapters import ChromaPrivGuard, LangChainPrivGuardEmbeddings

collection = ChromaPrivGuard.from_client(client, "notes", guard)
safe_embeddings = LangChainPrivGuardEmbeddings(provider, guard)
```

## CLI

```bash
# Simulate a dictionary inversion attack
privrag attack "patient diabetes ssn 12345"

# Benchmark retrieval retention
privrag benchmark --samples 500 --epsilon 1.0

# Sanitize an embedding batch
privrag protect --input raw.npy --output safe.npy --passphrase secret
```

## Security notes

This is a practical defense layer, not a formal end-to-end DP guarantee.
The bounded perturbation intentionally trades strict DP calibration for
retrieval utility. Use `DifferentialPrivacyEngine.add_noise()` for textbook
unbounded DP when formal guarantees are required.

Keep rotation keys outside the vector store. Apply one configuration to all
documents and queries in an index. Restrict database access. Encrypt at rest
and in transit.

## Development

```bash
pytest
```
