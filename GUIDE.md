# 🎧 Customer Support Router Agent — Full Deconstruction & Demo Guide

**Architecture & Implementation Deep Dive**  
**Core Patterns:** Autonomy Spectrum, Vector Store RAG, Confidence Routing, Deterministic Guardrails  
**Pipeline:** Perception → Memory (ChromaDB) → Reasoning → Action (Mock Execution & Escalation)

---

## 🎯 Why This Demo? (The Session 2 Connection)

This single demo covers **both** Session 2 objectives simultaneously:

| Session 2 Objective | How This Demo Covers It |
| :--- | :--- |
| **Point 1: AI Problem Deconstruction** | We deconstruct "build a customer support bot" from vague request → Wire-level components (Perception, Memory, Reasoning, Action) → State Schemas → Guardrails → Autonomy Dial |
| **Point 2: Rapid Prototyping Tools** | We run the actual working prototype live, showing Pydantic structured output, ChromaDB vector retrieval, confidence-based routing, and audit trails — all in ~200 lines of Python |

---

# 🧠 Part 1: What Does This Agent Look Like? (The Figurative View)

## The Mental Model: A Customer Support Desk with Rules

Imagine a human support agent sitting at a desk. They have:
- **Ears & Eyes** (Perception) — They read the customer message, detect mood, extract key details.
- **A Policy Binder on the Desk** (Memory / Knowledge Base) — They flip to the right policy page.
- **A Brain with Judgment** (Reasoning) — They compare the customer's request against the policy and decide: *"Can I handle this, or do I need a manager?"*
- **Hands & Phone** (Action) — They either resolve the issue or pick up the phone and escalate.

Now replace the human with an AI agent. The architecture is identical:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                    THE CUSTOMER SUPPORT AGENT (FIGURATIVE VIEW)                        │
│                                                                                        │
│   📩 Customer Message Arrives                                                          │
│          │                                                                             │
│          ▼                                                                             │
│   ┌─────────────────┐    "What does the customer want?"                                │
│   │  👂 PERCEPTION   │    • Intent Classification (refund, password_reset, billing)     │
│   │  (perception.py) │    • Sentiment Detection (neutral, angry, frustrated)            │
│   │                  │    • Entity Extraction (order_id, amount, email)                 │
│   └────────┬────────┘    • Urgency Scoring (low, medium, high)                         │
│            │                                                                           │
│            ▼                                                                           │
│   ┌─────────────────┐    "What do our rules say about this?"                           │
│   │  📚 MEMORY       │    • ChromaDB Vector Store (semantic search)                     │
│   │  (memory.py)     │    • Knowledge Base: refund-policy.md, escalation-rules.md,      │
│   │                  │      password-reset.md, billing-faq.md                           │
│   └────────┬────────┘    • Returns top-K most relevant policy documents                │
│            │                                                                           │
│            ▼                                                                           │
│   ┌─────────────────┐    "Can I handle this, or do I need a human?"                    │
│   │  🧠 REASONING    │    • Matches intent against retrieved policies                   │
│   │  (reasoning.py)  │    • Generates confidence score (0.0 to 1.0)                    │
│   │                  │    • ⚠️ POST-LLM GUARDRAIL: If confidence < 0.70, force escalate│
│   └────────┬────────┘    • Determines action_type: "resolve" or "escalate"             │
│            │                                                                           │
│            ▼                                                                           │
│   ┌─────────────────┐    "Execute the decision"                                        │
│   │  ⚡ ACTION        │    • ✅ Resolve: Mock API calls (Stripe refund, SendGrid email) │
│   │  (action.py)     │    • 🚨 Escalate: Create Zendesk ticket with full context       │
│   │                  │    • 📝 ALWAYS: Write to audit trail (episodic memory / JSONL)   │
│   └─────────────────┘                                                                  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

## The Autonomy Spectrum (The Graduated Dial)

This agent doesn't have a binary ON/OFF switch. It operates on a **graduated autonomy dial** — this is a foundational concept in production agent design:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          THE AUTONOMY SPECTRUM (3 ZONES)                               │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│  🟢 ZONE GREEN: Fully Autonomous                                                       │
│  ├── Password resets (clear match, low risk)                                           │
│  ├── FAQ answers (exact knowledge base match)                                          │
│  └── Small refunds ≤ $50 (within policy threshold)                                     │
│                                                                                        │
│  🟡 ZONE YELLOW: Supervised / Approval Required                                        │
│  ├── Medium refunds $50–$200 (agent proposes, human approves)                          │
│  └── Account changes (agent summarizes, human confirms)                                │
│                                                                                        │
│  🔴 ZONE RED: Mandatory Human Escalation                                               │
│  ├── Refunds > $200 (too high-risk for automation)                                     │
│  ├── Angry sentiment + legal threats                                                    │
│  ├── Unknown intents (no policy match)                                                 │
│  ├── Account deletion requests                                                         │
│  └── Agent confidence < 0.70 threshold                                                 │
│                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

> **Key Architectural Insight:** The agent's guardrail is *not* in the LLM prompt alone. There is a **hardcoded post-LLM confidence override** in `reasoning.py` (lines 77-80): even if the LLM says "resolve," if confidence is below 0.70 or limits are breached, the code forces escalation. This is the difference between an unverified toy demo and an enterprise-grade agent.

---

# ⚙️ Part 2: Key Technical & Functional Concepts

## Concept 1: Structured Output via Pydantic (Zero Raw Text Parsing)

The LLM never returns raw text that we regex-parse. Every response is forced into a **strict Pydantic schema**:

```python
# perception.py — The LLM MUST output this exact shape
class MessageIntent(BaseModel):
    intent: str       # "refund", "password_reset", "billing_question", "unknown"
    urgency: str      # "low", "medium", "high"
    sentiment: str    # "positive", "neutral", "negative", "angry"
    entities: ExtractedEntities  # { order_id, amount, email }

# reasoning.py — The decision MUST output this exact shape
class AgentDecision(BaseModel):
    action_type: Literal["resolve", "escalate"]
    confidence: float           # 0.0 to 1.0
    rationale: str              # Policy-grounded explanation
    action_parameters: dict     # { "amount": 30.0, "order_id": "9999" }
```

**Why this matters for an AI PM:**
- No ambiguous outputs. Engineering never debates "what did the model mean?"
- Testable: Every response is a typed object with known fields. You can write unit tests.
- Pydantic validates at parse time — if the LLM returns garbage, it throws an error instead of silently corrupting downstream logic.

---

## Concept 2: Semantic Memory (Vector Store = The Policy Binder)

The agent doesn't stuff all 4 policy documents into the prompt every time (that would waste tokens). Instead:

```
STARTUP: Load all .md files from knowledge-base/ into ChromaDB
         ChromaDB auto-embeds them using all-MiniLM-L6-v2 (local, free, fast)

RUNTIME: For each customer message:
         1. Perception extracts intent → "refund policy and rules"
         2. Memory.retrieve() runs semantic search against ChromaDB
         3. Returns top-2 most relevant policy chunks
         4. Only THOSE 2 documents enter the Reasoning prompt
```

**Why this matters for an AI Engineer & PM:**
- **Token Optimization:** Instead of stuffing ~2,000 tokens of all policies, we inject only ~400 tokens of relevant ones.
- **Scalability:** When you have 500+ policy documents across merchant tiers, you can't fit them all in context. Semantic retrieval is mandatory.
- **Accuracy:** The model reasons over the *right* policy instead of getting confused by irrelevant content.

---

## Concept 3: The Post-LLM Guardrail (Defense in Depth)

The LLM is asked to output a confidence score. But LLMs can be overconfident. So there's a **hardcoded programmatic override** after the LLM responds:

```python
# reasoning.py lines 77-80 — THE CRITICAL GUARDRAIL
if decision.action_type == "resolve" and decision.confidence < CONFIG["CONFIDENCE_THRESHOLD"]:
    print("⚠️ OVERRIDE: LLM suggested resolve, but confidence too low. Forcing escalation.")
    decision.action_type = "escalate"
```

**Why this matters for an AI Engineer & PM:**
- Never rely on the LLM alone for safety-critical decisions.
- **Deterministic code wraps probabilistic model** — this is the architecture pattern that separates production agents from demos.
- The threshold (0.70) is a **system-defined business constant**, not an opaque model artifact. You control this dial.

---

## Concept 4: Episodic Memory / Audit Trail (The Action Log)

Every single decision — resolve or escalate — gets written to a JSONL audit log:

```json
{
  "timestamp": "2026-08-25T23:30:00",
  "message": "I demand a refund of $150!",
  "parsed_intent": { "intent": "refund", "urgency": "high", "sentiment": "angry" },
  "decision": { "action_type": "escalate", "confidence": 0.65, "rationale": "..." }
}
```

**Why this matters for an AI Engineer & PM:**
- **Regulatory Compliance:** For financial and mission-critical products, you need an auditable trail of why the AI made each decision.
- **Evaluation / Golden Datasets:** These logs become your eval data. You can replay 100 past decisions and ask: "Would prompt v2 have made better decisions than v1?"
- **Debugging:** When an edge case occurs, you trace the exact perception → memory retrieval → reasoning chain.

---

## Concept 5: The 4-Component Architecture (Agent Framework)

This project is a textbook implementation of the 4-component agent model:

| Component | File | What It Does | Failure Mode to Watch |
| :--- | :--- | :--- | :--- |
| **Perception** | `perception.py` | LLM-powered intent classifier + entity extractor (Pydantic structured output) | Misclassifies "delete my account" as "billing_question" |
| **Memory** | `memory.py` | ChromaDB vector store with 4 knowledge base articles, semantic retrieval | Retrieves wrong policy (e.g., returns billing FAQ for a refund request) |
| **Reasoning** | `reasoning.py` | Policy-grounded decision engine with confidence scoring + post-LLM guardrail | Overconfident autonomy: resolves when it should escalate |
| **Action** | `action.py` | Mock API execution (Stripe/SendGrid) + mandatory JSONL audit trail | Executes refund but forgets to log it (audit gap) |

> **Debugging Rule of Thumb:** When an agent misbehaves, debug from the **outside in**: check Perception first → then Memory → then Reasoning → then Action. The model is almost never the problem.

---

# 🏗️ Part 3: How to Build This in Antigravity IDE

## Step-by-Step: Building the Support Router in Antigravity

### Prerequisites
```bash
# 1. Set your Gemini API key
export GEMINI_API_KEY="your-api-key-here"

# 2. Install dependencies
pip install google-genai chromadb pydantic python-dotenv
```

### Method 1: Build It with Antigravity CLI (`agy`)

Open Antigravity CLI in the demo directory and use natural language to scaffold:

```bash
cd customer-support-agent-demo

# Launch Antigravity CLI
agy
```

Then prompt Antigravity with:

```
Build a customer support agent with these 4 modules:

1. perception.py — Takes a customer message string, calls Gemini with Pydantic 
   structured output to extract: intent (refund/password_reset/billing/unknown), 
   urgency, sentiment, and entities (order_id, amount, email).

2. memory.py — Loads markdown files from knowledge-base/ directory into ChromaDB 
   on startup. Provides a retrieve(query, n_results=2) method for semantic search.

3. reasoning.py — Takes the parsed intent + retrieved policies, asks Gemini to 
   decide "resolve" or "escalate" with a confidence score. Has a hardcoded 
   post-LLM guardrail: if confidence < 0.70, force escalation regardless.

4. action.py — Executes the decision (mock Stripe refund / SendGrid email for 
   resolve, mock Zendesk ticket for escalate). Writes every decision to an 
   audit_trail.jsonl file.

The knowledge base has 4 files: refund-policy.md, escalation-rules.md, 
password-reset.md, billing-faq.md.
```

### Method 2: Run the Existing Code Directly

Run the agent directly from the repository root:

```bash
cd customer-support-agent-demo

# Copy environment variables template and configure your API key
cp .env.example .env
# Edit .env with your GEMINI_API_KEY

# Install dependencies
pip install -r requirements.txt

# Run the agent pipeline
python -m src.main
```

### Method 3: Use Antigravity IDE for Interactive Exploration

1. **Open the project** in Antigravity IDE
2. **Select any source file** (e.g., `reasoning.py`) and ask in the sidebar chat:
   - *"Walk me through how the confidence guardrail works in this file"*
   - *"What happens if I change CONFIDENCE_THRESHOLD from 0.70 to 0.90?"*
   - *"Add a new test case: a customer threatening legal action over a $25 refund"*
3. **Use `/plan` slash command** to have Antigravity design modifications:
   - *"Add a new knowledge base article for shipping delays and add a test case for it"*
   - *"Add input sanitization to detect prompt injection attempts before the LLM call"*

---

# 🎬 Part 4: The Live Demo Walkthrough (15-Minute Architecture Tour)

## The 6 Test Cases (Pre-Built in `main.py`)

| # | Test Message | Expected Behavior | What It Teaches |
| :--- | :--- | :--- | :--- |
| 1 | *"I forgot my password for john.doe@example.com"* | 🟢 **Auto-Resolve** — Password reset via SendGrid. High confidence, low risk. | Happy path. Agent handles autonomously. |
| 2 | *"I need a refund for order #9999, $30"* | 🟢 **Auto-Resolve** — Refund ≤ $50 threshold. Stripe mock API. | Demonstrates policy-bounded autonomy. |
| 3 | *"I demand a refund! Order #8888, $150, terrible service!"* | 🟡 **Escalate** — Amount $50–$200, angry sentiment. | Autonomy dial shifts: amount + sentiment = escalation. |
| 4 | *"Refund order #7777, $300"* | 🔴 **Mandatory Escalate** — Amount > $200, hard policy boundary. | Hard guardrail: no ambiguity, no model discretion. |
| 5 | *"How do I update my credit card?"* | 🟢 **Auto-Resolve** — FAQ answer, agent directs to dashboard. | Read-only response from knowledge base. |
| 6 | *"Delete my account and all my data right now."* | 🔴 **Escalate** — Account deletion = high-risk irreversible action. | Demonstrates "Forbidden Actions" concept. |

## Engineering Walkthrough Script

### Opening (2 mins)
> *"Imagine an enterprise receiving 500+ customer support tickets daily. The operations team is drowning. The CEO asks for an AI support agent. Instead of jumping into naive prompting, let's first deconstruct the problem into wire-level components, state schemas, and graduated safety boundaries."*

### Stage 1: Deconstruct the Problem (5 mins)
Walk through the 4-component architectural diagram on screen. Key engineering considerations:
1. *"If the agent incorrectly issues an automated refund, what is the blast radius?"*
2. *"Which decisions can be executed autonomously, and which require human approval?"*
3. *"Should business rules and calculations be delegated to probabilistic LLMs or deterministic software?"* (Layered safety principle)

### Stage 2: Run the Live Agent (5 mins)
Run the agent and examine each test case:
```bash
python -m src.main
```
For each output, observe:
- *Confidence scores and policy references in the agent rationale.*
- *How tuning `CONFIDENCE_THRESHOLD` immediately tightens or relaxes agent autonomy.*

### Stage 3: Examine the Wire (3 mins)
Open `reasoning.py` and inspect lines 77-80 (the post-LLM guardrail):
> *"Notice the post-LLM safeguard in `reasoning.py`. Even if the LLM suggests 'resolve', code deterministically forces 'escalate' if confidence is below threshold or monetary limits are exceeded. This is layered defense-in-depth: LLMs propose, deterministic software enforces."*

Open `action.py` and inspect the audit trail:
> *"Every single decision — resolve or escalate — is persisted in a structured JSONL log. In production environments, this audit trail satisfies regulatory compliance and serves as the continuous evaluation dataset for prompt regressions."*

---

# 📊 Part 5: Engineering Cheat Sheet — Concepts to Internalize

| Concept | Core Architectural Principle | Where in Code |
| :--- | :--- | :--- |
| **Structured Output (Pydantic)** | LLM never returns raw unvalidated text. Every response conforms to a typed schema. | `perception.py:11-15`, `reasoning.py:8-21` |
| **Semantic Memory (Vector Store)** | Prevent context window bloat. Embed at startup, retrieve top-K at runtime. | `memory.py:26-61` |
| **Post-LLM Guardrail** | Deterministic code wraps probabilistic model. Safety thresholds are explicit system constants. | `reasoning.py:77-80` |
| **Autonomy Spectrum (3 Zones)** | Autonomy is graduated. Green/Yellow/Red zones defined by dollar amount, sentiment, and confidence score. | `config.py:19-20`, `escalation-rules.md` |
| **Episodic Memory (Audit Trail)** | Every decision logged as JSONL. Enables compliance, debugging, and eval dataset creation. | `action.py:51-68` |
| **Debug Order** | Perception → Memory → Reasoning → Action. The model is almost never the problem. | Architecture pattern |
| **Temperature Tuning** | Classification/Routing = 0.1 (deterministic). Reasoning = 0.2 (slightly creative for rationale writing). | `perception.py:45`, `reasoning.py:70` |

---

# 🔗 Part 6: File Reference & Directory Structure

```
customer-support-agent-demo/
├── GUIDE.md                         ← You are here (this file)
├── requirements.txt                 ← Dependencies: google-genai, chromadb, pydantic
├── knowledge-base/                  ← The agent's "policy binder" (semantic memory source)
│   ├── refund-policy.md             ← Refund thresholds: ≤$50 auto, $50-$200 approval, >$200 escalate
│   ├── escalation-rules.md          ← When to escalate: high risk, low confidence, angry, unknown
│   ├── password-reset.md            ← Password reset procedure and limitations
│   └── billing-faq.md               ← Billing questions and payment method rules
├── src/                             ← The 4-component agent implementation
│   ├── config.py                    ← Business constants: API key, model, thresholds
│   ├── perception.py                ← 👂 Intent classifier + entity extractor (Pydantic + Gemini)
│   ├── memory.py                    ← 📚 ChromaDB vector store loader + semantic retriever
│   ├── reasoning.py                 ← 🧠 Decision engine + confidence guardrail
│   ├── action.py                    ← ⚡ Mock API executor + audit trail logger
│   └── main.py                      ← 🎬 Orchestrator: runs 6 test cases through the pipeline
└── logs/                            ← Auto-created at runtime
    └── audit_trail.jsonl            ← Every decision logged (episodic memory)
```
