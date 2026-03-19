# Citation Compass integration plan for Hyrax

## Purpose
Implement a **minimal, maintainable** citation workflow that covers both:

1. **Hyrax as a dependency** (downstream projects can cite Hyrax itself).
2. **Hyrax dependencies** (Hyrax can emit citations for key method-defining packages).

This plan is intentionally short in scope and aligned with Hyrax design principles in `HYRAX_GUIDE.md` (simple defaults, single obvious workflow, low user burden).

---

## What I reviewed before proposing this

- Project architecture and assistant guidance: `HYRAX_GUIDE.md`, `CLAUDE.md`, `.github/copilot-instructions.md`.
- Packaging + dependency surface: `pyproject.toml`.
- Current CLI architecture and extension pattern: `src/hyrax_cli/main.py`, `src/hyrax/hyrax.py`, `src/hyrax/verbs/*`.
- Docs structure and discoverability: `docs/index.rst`, `README.md`, docs reference pages.

This matters because the integration should fit existing Hyrax patterns (Jupyter first, CLI second, avoid unnecessary new config knobs/verbs unless justified).

---

## Scope for first implementation

### In scope
- Add authoritative citation metadata for Hyrax itself.
- Add Citation Compass config/data for a **small curated set** of high-impact runtime dependencies.
- Provide **one canonical way** for users/maintainers to generate citations.
- Add lightweight tests/validation and contributor documentation.

### Out of scope
- Auto-citing all transitive dependencies.
- Workflow-specific citation graph logic.
- Introducing complex policy/configuration UI in this first pass.

---

## Minimal implementation steps (with concrete file targets)

### 1) Add Hyrax software citation metadata (`CITATION.cff`)

**Create:** `CITATION.cff` at repo root.

**Include:**
- `cff-version`
- `title` (`Hyrax`)
- `message`
- `authors` (LINCC Frameworks + maintainers as appropriate)
- `repository-code` (`https://github.com/lincc-frameworks/hyrax`)
- `license` (`MIT`)
- `version` strategy tied to releases (manual update or release automation)
- `date-released` on release tags
- `doi` if/when minted

**Why:** This is the standard artifact GitHub and downstream users expect for citing software.

---

### 2) Add Citation Compass dependency source file

**Create (name per tool convention once verified):** one root-level Citation Compass config/source file.

**Initial curated dependency set (from `pyproject.toml`):**
- `torch`
- `pytorch-ignite`
- `astropy`
- `mlflow`
- `umap-learn`
- `lancedb`
- `pyarrow`
- (optionally) one vector DB backend package actively documented in workflows (`chromadb` or `qdrant-client`)

**Selection rule for v1:** include only dependencies that are methodologically central to Hyrax scientific outcomes, not utilities.

**Why:** keeps maintenance small while covering likely citation requirements.

---

### 3) Add one canonical citation generation path

Pick exactly one user-facing interface for v1:

- **Preferred (lowest code risk):** document a command based on Citation Compass CLI/module invocation.
- **Optional follow-up:** add a dedicated `hyrax cite` command only if maintainers want tighter UX parity with existing verbs.

**Initial recommendation:** start documented-only (no new Hyrax verb yet), then reassess after real user feedback.

**Why:** consistent with “make easy things easy, hard things possible” while avoiding premature CLI surface expansion.

---

### 4) Document usage + maintenance policy

**Update docs:**
- `README.md` (brief “How to cite Hyrax and dependencies” section)
- `docs/reference_and_faq.rst` or another stable reference page
- Developer/contributor guidance with a short policy snippet:
  - add dependency citations only for methodologically central libs,
  - review citation list during release prep,
  - keep `CITATION.cff` current.

**Why:** without policy, citation lists sprawl quickly.

---

### 5) Add basic validation checks

**Add tests/checks with minimal friction:**
- A packaging/documentation test ensuring `CITATION.cff` exists and has required top-level fields.
- A check that Citation Compass config file is present and parseable.
- Optional: smoke test for citation command execution if dependencies are available in CI.

Likely location: `tests/hyrax/test_packaging.py` (or nearby packaging/docs test module).

**Why:** prevents silent regressions and missing citation metadata in releases.

---

## Suggested sequence for implementation

1. **Merged PR A+B:** add `CITATION.cff`, add Citation Compass config, add docs usage snippet, and add lightweight validation checks in one implementation PR.

No `hyrax cite` PR is planned for this phase.

---

## Decision log for future implementer

- **Why no immediate new verb?** Hyrax guidance recommends avoiding new verbs unless clearly needed; documented command is enough for v1.
- **Why not all dependencies?** Most packages in `pyproject.toml` are infrastructure/utilities; citing all by default increases noise and maintenance burden.
- **Why root-level plan and files?** Citation metadata and citation tooling are repository-level concerns, not core scientific docs pages.

---

## Acceptance criteria (v1 done)

- `CITATION.cff` exists and is valid enough for GitHub citation UI.
- `citation_compass.toml` exists and includes a curated dependency list.
- One canonical documented command/path exists for generating citations.
- Docs clearly explain when to update citation metadata and dependency entries.
- Automated checks fail if key citation artifacts are removed or malformed.

---

## Risks and mitigations

- **Risk:** dependency citation entries become stale.
  - **Mitigation:** add “citation review” to release checklist and a simple test for file presence/shape.
- **Risk:** command UX confusion if multiple ways are documented.
  - **Mitigation:** explicitly mark one canonical path.
- **Risk:** overgrowth of citation file.
  - **Mitigation:** enforce “methodologically central only” policy in contributor docs.
