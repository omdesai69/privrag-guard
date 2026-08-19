# Universal Engineering Rulebook (v6 — Production-Grade & Token-Lean)

This rulebook defines the development standard for all software engineering projects (Full-Stack, Backend, Systems, AI/ML, CLI, Cloud). It enforces senior Google-level craftsmanship, maximum security, optimal algorithmic complexity, and extreme token efficiency.

---

## 1. Documentation Policy (Zero File Clutter & Token Optimization)
* **Standard Project Documentation**:
  - `README.md` — Clear installation, configuration, public API, and quickstart commands.
  - `ARCHITECTURE.md` — In-depth technical design, component diagrams (Mermaid), stack rationale, and end-to-end data flow.
  - `SECURITY.md` — Created only when handling authentication, cryptographic key management, or sensitive PII/compliance boundaries.
* **Prohibited File Bloat**:
  - **Never** generate throwaway markdown files (`VISION.md`, `API.md`, `COST.md`, `DATABASE.md`, `DONE.md`, `SCALABILITY.md`, `THREAT_MODEL.md`, `TODO.md`, `ADR/`) across any project.
  - **Internalized Principles**: Threat modeling (STRIDE), scalability bottlenecks, cost curves, database migrations, and API design must be actively thought through and applied **directly in the architecture, code, and tests** without creating separate markdown files.

---

## 2. Code Craftsmanship & The 500-Line Conciseness Lens
* **Senior Engineering Self-Critique**:
  - Before finalizing any file, ask yourself 4 to 5 times: *"Can this 1,000 lines of code be engineered in 400–500 lines of clean, idiomatic code without losing readability or correctness?"*
  - Ruthlessly eliminate boilerplate, redundant wrappers, AI-templated filler, and duplicate helper functions.
* **Algorithmic Time & Space Complexity**:
  - Analyze Big-O complexity for every core path ($O(1) / O(\log n) / O(n)$ preferred; strictly flag and justify anything $O(n^2)$ or worse).
  - Vectorize operations (NumPy, BLAS, SIMD) and eliminate nested loops.
  - Optimize memory footprint: stream large datasets, use generators/iterators, avoid duplicate in-memory state.
* **Dependency Minimization**:
  - Prioritize standard libraries and lightweight, maintained primitives over heavy, bloated frameworks.

---

## 3. Maximum Security by Default (Zero-Trust Standard)
* **Secrets & Credentials**: Never hardcode secrets or commit `.env` files. Retrieve via environment variables or secret managers.
* **Injection Defense**: 100% parameterized queries / ORMs; zero string-concatenated SQL, shell, or query builders.
* **Cryptographic Standards**: Use proven, memory-hard algorithms (`scrypt`, `argon2id`, AES-GCM, ed25519); never hand-roll cryptography.
* **Input Validation & Memory Safety**: Strict schema validation (Zod, Pydantic, Joi) at system boundaries, explicit type hints, NaN/Inf checks, and safe deserialization (`allow_pickle=False`, `safe_load`).
* **Abuse Protection**: Implement rate limiting, timeouts, circuit breakers, and bounded payloads across all network and API endpoints.

---

## 4. Testing & Terminal Verification Gate
* Every library, service, or CLI tool must include comprehensive automated tests (unit + integration).
* Maintain $\ge 90\%$ test coverage.
* **Mandatory Terminal Verification**: Always execute real terminal commands and test suites (`pytest`, `npm test`, `cargo test`, CLI runs) to verify live correctness before completing a task.

---

## 5. Communication & Turn Handoffs
* Keep responses concise, direct, and focused on verified results.
* No repetitive robotic boilerplate blocks on trivial turns. State assumptions or questions only when user input is strictly required.
