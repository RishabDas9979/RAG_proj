# Compliance AI Suite: PolicyGuard + BankComply

Two retrieval-grounded compliance tools. Everything runs on the Python standard library plus
`numpy` and `scikit-learn`; no API key is required.

| Project | What it does |
|---|---|
| **PolicyGuard** | Checks a policy document against a requirements framework. Finds missing controls, vague wording, placeholders and internal conflicts, and answers questions with citations. |
| **BankComply** | A KYC/AML case agent. Screens customers, runs transaction-monitoring rules, scores risk with an explanation, cites the bank's own procedure, and runs cases through a role-controlled, four-eyes workflow with a tamper-evident audit log. |

BankComply reuses PolicyGuard's hybrid retriever to cite the operating procedure behind each decision.

## Quick start

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -t .          # 76 tests

python -m policyguard check data/policies/weak_policy.md
python -m policyguard check data/policies/good_policy.md --fail-under 80
python -m policyguard ask data/policies/good_policy.md "How quickly must a breach be reported?"

python -m bankcomply.demo                          # investigate 200 customers, evaluate, approve a SAR
python -m bankcomply.api                           # JSON API on :8000 (demo tokens)
```

## PolicyGuard

```
policy.md -> structure-aware chunking -> hybrid index (BM25 + TF-IDF, rank fusion)
                                              |
        framework JSON (requirements) -> evidence per requirement -> met / partial / missing
                                              |
                  wording lint (vague terms), placeholders, conflicting retention periods
```

- **Chunking** splits on headings and keeps section numbers, so citations read like "section 8, Incident Response".
- **Retrieval** fuses BM25 and TF-IDF cosine with reciprocal rank fusion. Acronyms such as MFA, KYC and PEP are expanded on the query side.
- **Checks** use a requirements framework in `policyguard/frameworks/` (16 controls). A requirement is `met` only if the policy names an owner, a frequency or a time limit where the framework asks for one. Otherwise it is `partial` with the gap spelled out.
- **Q&A** is extractive and abstains when nothing relevant is found. An optional `llm=` callable can generate answers, and every generated sentence is checked against the retrieved clauses.

Sample results: `good_policy.md` scores Strong; `weak_policy.md` scores Weak (missing MFA, sanctions screening, continuity and vendor controls, plus a 5 vs 7 year retention conflict and two TBD placeholders).

## BankComply

```
customer -> KYC checks -> watchlist screening -> transactions -> AML rules
                                                                    |
              policy lookup (PolicyGuard) <- explainable risk score <-
                       |
              narrative -> case (only if risk) -> analyst -> officer (four-eyes) -> SAR approved
                       |
        every tool call and decision -> hash-chained audit log
```

- **Rules**: structuring, rapid movement, velocity vs. the customer's profile, high-risk jurisdictions, large cash. Each alert lists the transactions behind it.
- **Scoring**: every point is attributed to a named reason. The agent *recommends*; it never approves a SAR.
- **RBAC**: analyst, senior analyst, compliance officer, auditor, admin. Auditors see masked names; admins see the audit log but no case data (separation of duties).
- **Workflow**: states and notes are enforced, and the SAR approver must differ from the person who prepared and escalated the case.
- **Audit log**: SQLite, append-only triggers, each row hash-chained to the previous one. Denied and rejected attempts are logged too.
- **Narrative guard**: an optional model-written narrative is rejected if it contains any figure absent from the evidence.

### Results (read these carefully)

All numbers come from the **synthetic** generator in `bankcomply/data.py`, whose typologies I wrote
together with the rules. They show the plumbing works, not how well it would detect real laundering.

| Dataset (200 customers, 12 should-be-SAR) | SAR precision | SAR recall | Any-alert recall | False alerts on clean customers |
|---|---|---|---|---|
| Plain: typologies sit on the rule thresholds | 100% | 100% | 100% | 0 |
| Evasive: typologies avoid exact thresholds | 100% | **33%** | 100% | 0 |

In the evasive run, 8 of 12 SAR-worthy customers are downgraded to enhanced due diligence instead of a
SAR recommendation. They are still flagged for human review, but it shows the fixed thresholds are easy to
dodge. The natural next step is a learned anomaly model next to the rules.

## Data

- `bankcomply.data.generate_dataset()` builds customers, transactions and ground-truth labels with injected typologies and hard negatives (benign large wires, PEPs, expired documents, near-miss names).
- `bankcomply.data.load_paysim_csv(path)` reads Kaggle's PaySim layout. **It has only been tested on a tiny file in that layout**: Kaggle could not be reached from the build environment, so it has not been run on the real dataset. PaySim also has no names, countries or PEP flags, so most rules and screening will not apply to it.
- Watchlist names, "high-risk" country codes (`XA`, `XB`, `XC`) and both policies are fictional placeholders. Replace them with real lists and your institution's actual obligations before any real use.

## Limits

- Not legal or regulatory advice; not a production AML system.
- Retrieval is lexical (BM25 + TF-IDF), so unusual paraphrases can be missed; embeddings would be the next step.
- The audit chain detects edits and deletions, but someone who can rewrite the whole database could rebuild a consistent chain. Store `AuditLog.head()` somewhere else and pass it to `verify(expected_head=...)`.
- The API uses static demo tokens; use a real identity provider in production.

## Layout

```
policyguard/   chunking, retrieval, checker, qa, cli, frameworks/
bankcomply/    data, aml_rules, kyc, scoring, rbac, audit, cases, agent, api, evaluate, demo
data/policies/ sample policies and the AML operating procedure
tests/         unittest suites for both projects
```
