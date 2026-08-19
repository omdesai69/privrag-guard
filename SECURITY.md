# Security Controls & Policy

## 1. Cryptographic Key Management
- **Key Derivation**: All rotation matrices derived from passphrases use `hashlib.scrypt` with $N=16384$, $r=8$, $p=1$, and domain-separated salts.
- **Key Storage**: Never commit rotation keys or `.env` files to git. Key files written with `guard.save_key()` should reside on memory-backed filesystems (`tmpfs`) or secret volumes with `chmod 600`.
- **Pickle Prohibition**: Key loading explicitly enforces `allow_pickle=False` in `np.load` to eliminate arbitrary code execution vectors.

## 2. Input Validation & Memory Safety
- **Finite Checking**: All incoming matrices and vectors undergo strict `np.isfinite(vector).all()` assertions to prevent NaN/Inf poisoning.
- **Dimension Consistency**: Guard instances strictly lock to the dimension of the first protected vector ($d \times d$ rotation matrix). Any mismatched dimension raises an immediate `ValueError`.
- **Zero-Vector Protection**: Zero vectors are rejected at the boundary to avoid division-by-zero singularities during normalization.

## 3. Defense-in-Depth Limitations
- **Not Homomorphic Encryption**: PrivRAG-Guard is a lightweight defense-in-depth transformation layer. It dramatically increases the cost of dictionary and gradient-based inversion attacks while preserving standard vector DB cosine query performance.
- **Single Index Consistency**: All documents and queries in an index MUST be protected with the exact same guard key and configuration.
