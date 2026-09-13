# SpotBot — AI Support Agent for SpotifyCares

An AI-powered customer support agent built from real Twitter support conversations. The agent classifies customer intents, retrieves relevant knowledge, and drafts replies grounded in how SpotifyCares has historically resolved similar issues.

**Brand**: SpotifyCares (selected from 108 brands in the dataset)
**Dataset**: Customer Support on Twitter (~2.8M tweets)
**LLM**: Groq API (qwen3.8-27b)
**Retrieval**: Semantic embeddings (sentence-transformers) + ChromaDB

---

## Quick Start (Reproduce Results in Under 15 Minutes)

### Prerequisites

- Python 3.10+
- ~500MB disk space
- Internet connection (for Groq API + model download on first run)

### Setup

```bash
# Clone and enter the repo
cd spotify_bot

# Install dependencies
pip install scikit-learn numpy groq sentence-transformers chromadb python-dotenv openai

# Create .env file with your API key
echo "GROQ_API_KEY=your_groq_api_key_here" > .env

# Place the dataset
# Download twcs.csv from Kaggle (thoughtvector/customer-support-on-twitter)
# Place it at ../twcs.csv (one directory above spotify_bot/)
```

### Run the Pipeline

```bash
# Build everything (takes ~5 minutes)
python3 scripts/01_extract_conversations.py     # Extract 26K SpotifyCares conversations
python3 scripts/02_label_intents.py             # Label intents (keyword bootstrap)
python3 scripts/03_build_knowledge_base.py      # Build KB from scraped articles
python3 scripts/04_train_intent_classifier.py   # Train baseline classifier (84.5%)
python3 scripts/05_build_rag_pipeline.py        # Embed KB into ChromaDB
python3 scripts/07_extract_historical_responses.py  # Mine 856 resolved conversations
python3 scripts/08_embed_historical.py          # Embed historical into ChromaDB
```

### Launch the Bot

```bash
python3 bot.py
```

### Run Evaluation

```bash
# Run evaluation harness (50 sampled golden examples, ~10 min)
python3 scripts/09_evaluation_harness.py
```

---

## Report

### 1. Problem Framing

**What "good" means for SpotifyCares:**

A good support agent for Spotify must do three things:
1. **Correctly identify the issue** — "shuffle plays same songs" is about shuffle settings, not a general playback bug
2. **Give actionable next steps** — not "we're sorry" or "DM us", but actual fix steps or a targeted diagnostic question
3. **Follow the diagnostic ladder** — real agents don't dump all steps at once. They ask "what device?" first, then give the right fix for that device

We defined "good" by studying how SpotifyCares agents actually behave in their best conversations (the 856 where the customer confirmed "that worked, thanks!").

**What we chose NOT to build:**

- **Account-specific actions** (checking billing, resetting passwords, issuing refunds) — the bot redirects these to human support since they require private account access
- **Multi-language support** — 99.8% of SpotifyCares conversations are English; non-English was excluded
- **Proactive outreach** — the bot only responds to incoming queries, doesn't initiate contact
- **Real-time account integration** — the bot cannot query Spotify's internal systems (billing, account status, password resets) in real-time

---

### 2. Results vs. Baselines

We tested three approaches for intent classification and two for retrieval:

#### Intent Classification

| Approach | Type | Accuracy | Notes |
|---|---|---|---|
| Keyword matching | Trivial baseline | ~75% coverage | 24.5% of tweets couldn't be labeled at all |
| TF-IDF + Logistic Regression | Simple baseline | 84.5% (3,935 test samples) | Confused "crashing on iphone" as device_compatibility |
| **LLM (Groq qwen3.8-27b)** | **Final system** | **84% on 50 golden examples** | Handles paraphrasing and edge cases |

Per-intent breakdown for the simple baseline (TF-IDF + LR):

| Intent | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| subscription_billing | 0.95 | 0.89 | 0.92 | 1,145 |
| content_availability | 0.86 | 0.89 | 0.87 | 176 |
| device_compatibility | 0.89 | 0.81 | 0.85 | 1,575 |
| shuffle_queue | 0.83 | 0.88 | 0.85 | 82 |
| account_access | 0.81 | 0.87 | 0.84 | 225 |
| download_offline | 0.83 | 0.86 | 0.84 | 167 |
| playback_issue | 0.69 | 0.80 | 0.74 | 144 |
| ads_complaints | 0.71 | 0.76 | 0.73 | 62 |
| playlist_library | 0.58 | 0.88 | 0.70 | 256 |
| feature_request | 0.49 | 0.76 | 0.60 | 51 |
| audio_quality | 0.58 | 0.48 | 0.53 | 31 |
| country_region | 0.70 | 0.76 | 0.73 | 21 |

#### KB Retrieval

| Approach | Direct queries | Paraphrased queries | Avg score |
|---|---|---|---|
| TF-IDF cosine similarity | 10/10 (100%) | ~50% (fails without keywords) | 0.198 |
| **Semantic embeddings + ChromaDB** | **6/6 (100%)** | **6/6 (100%)** | **0.432** |

Key paraphrased queries that TF-IDF failed on but embeddings handled:

| Query (no KB keywords) | Correct article | Embedding score |
|---|---|---|
| "wireless speaker won't pair with my phone" | Bluetooth | 0.416 |
| "took money from my bank but I cancelled" | Failed Payment | 0.285 |
| "it keeps repeating the same tracks" | Shuffle | 0.437 |
| "songs sound muffled and distorted" | Audio Quality | 0.418 |
| "my saved music for the plane is all gone" | Downloads Removed | 0.296 |

#### Response Quality (V3 KB-only vs V4 KB+Historical)

| Aspect | V3 (KB only) | V4 (KB + Historical) |
|---|---|---|
| Response style | Dumps all steps at once | Asks diagnostic question first, like real agents |
| Grounded in historical data? | No | Yes — uses 223 resolution-confirmed conversations |
| Tone | Helpful but generic | Matches actual SpotifyCares agent tone |

---

### 3. Failure Analysis: Top 5 Failure Modes

#### Failure 1: `device_compatibility` absorbs everything (TF-IDF classifier)

**Example**: "spotify keeps crashing on my iphone"
**Expected**: `playback_issue` | **Got**: `device_compatibility`
**Why**: The word "iphone" triggers device keywords. TF-IDF can't distinguish "crashing ON an iphone" (playback) from "connecting TO an iphone" (device).
**Frequency**: ~15% of playback_issue tweets misclassified in TF-IDF baseline.
**Fix**: LLM classifier understands the semantic difference.

#### Failure 2: No KB article for Discover Weekly / Daily Mix

**Example**: "my discover weekly is giving terrible recommendations"
**Expected retrieval**: playlist_library article | **Got**: Ads article (wrong)
**Why**: We have no dedicated article for personalized playlists in the KB. The retriever picks the closest match, which is wrong.
**Frequency**: Affects all playlist recommendation queries.
**Fix**: Add a dedicated article on algorithmic playlists to the KB.

#### Failure 3: Sparse historical data for some intents

**Example**: `audio_quality` has only 5 resolved conversations, `country_region` has 9.
**Impact**: The bot's diagnostic flow for these intents is less natural because it has fewer real agent patterns to learn from.
**Why**: These issues are rare in the tweet data, or they get resolved via DM (not publicly).
**Fix**: Supplement with synthetic examples or community forum data.

#### Failure 4: Bot can't handle multi-intent queries

**Example**: "my app crashes AND I was charged twice"
**Expected**: Both `playback_issue` and `subscription_billing`
**Got**: Whichever the LLM picks first (usually billing due to stronger signal)
**Why**: The classifier outputs exactly one intent. No multi-label support.
**Fix**: Allow multi-label classification, retrieve KB for both intents.

#### Failure 5: Historical data teaches "ask first" even when the answer is obvious

**Example**: "how much does the family plan cost"
**Best response**: "$21.99/month" (immediate answer from KB)
**Actual response sometimes**: "What country is your account registered in?" (diagnostic question)
**Why**: Historical conversations almost always start with a question. The LLM over-indexes on this pattern for questions that have a direct factual answer.
**Fix**: Add a rule: if the KB has a direct factual answer and the query is a simple question (not a complaint/issue), answer directly.

---

### 4. "What Is Misleading About My Headline Number?"

**Headline: 84% intent accuracy, 4.05/5 response quality, Kappa 0.709.**

These are our real numbers from the golden evaluation set — but they're still misleading:

1. **50 examples is small.** We evaluated on 50 randomly sampled golden tweets (out of 200 labeled). Per-intent F1 scores for rare intents (ads_complaints: n=1, audio_quality: n=0) are statistically meaningless. The 84% could swing ±10% on a different 50-sample draw.

2. **We defined the intents, then tested on them.** The 12 categories were designed by us from the data. There's circular logic: we defined what "correct" means, then tested whether we match our own definition. Real users may have intents we didn't define (the 24.5% "other" category is proof of this).

3. **The LLM judges its own kind.** Our LLM-as-judge (qwen3.8-27b) scored responses generated by the same model family. LLMs tend to rate LLM-generated text more favorably than humans do. The Kappa of 0.709 is substantial but not perfect — the judge scored tone at 4.74 while the human scored it 4.70 (close), but on factual accuracy the gap is larger (3.80 vs 4.02 human avg).

4. **Escalation accuracy is 34%.** The bot includes "contact support" links in almost every response, inflating the "predicted escalate" count. This makes the headline intent accuracy look better than the end-to-end experience — the bot might get the intent right but still give a suboptimal response by escalating when it shouldn't.

5. **No evaluation on real user satisfaction.** We never tested with actual users. A response that looks correct to us might not satisfy a frustrated customer. There's no thumbs-up/thumbs-down data.

**The most defensible number**: 84% intent accuracy on 50 hand-labeled golden examples, with response quality of 4.05/5 as scored by an LLM judge validated at Kappa 0.709 against human scoring.

---

### 5. What We'd Do With One More Week

*Items 1-3 were originally "next steps" but we implemented them during development:*
- ~~Golden evaluation set~~ → Done (200 hand-labeled, Section 7)
- ~~LLM-as-judge evaluation~~ → Done (Kappa 0.709, Section 8)
- ~~Escalation logic~~ → Done (urgency + turn-based, Section 9)

**Remaining improvements:**

1. **Auto-updating KB scraper** — Build a scheduled scraper that monitors Spotify's help center (support.spotify.com) for new/updated articles and automatically re-embeds them into ChromaDB. The current KB is a static snapshot — Spotify updates pricing, features, and troubleshooting guides regularly. A weekly cron job scraping the help center + community forums would keep the KB fresh without manual intervention.

2. **Customer feedback loop** — Add a thumbs-up/thumbs-down rating after every bot response. Low-rated conversations get flagged and reviewed:
   - An LLM analyzes each low-rated conversation to identify WHY it failed (wrong intent? bad KB match? unhelpful response?)
   - Failed conversations with corrected labels get added back to the training set
   - High-rated conversations get added to the historical responses collection, continuously improving the bot's tone and diagnostic patterns
   - This creates a self-improving flywheel: more conversations → better training data → better responses → more positive ratings.

3. **Fix escalation accuracy (currently 34%)** — The bot over-escalates by including "contact support" links even when it can resolve the issue. Need to tighten the escalation threshold so it only escalates when it genuinely can't help.

4. **Fine-tune a small classifier** — Replace the LLM intent call with a fine-tuned DistilBERT on our labeled data. Same accuracy, 100x cheaper, 10x faster. Keep the LLM only for response generation.

5. **Multi-intent support** — Allow the classifier to output 2 intents for compound queries like "my app crashes AND I was charged twice".

6. **Evaluate on full 200 golden examples** — We evaluated on 50 due to API rate limits. Running on all 200 would give more reliable per-intent F1 scores.

---

### 6. Decision Log

| # | Decision | Why |
|---|---|---|
| 1 | **Picked SpotifyCares over AmazonHelp** | Amazon has 4x more data but almost zero public resolution. Spotify had the highest resolution rate (33.5%) of all 108 brands — agents give actual steps, not just "DM us". |
| 2 | **Defined 12 intents manually, not via clustering** | Unsupervised clustering on noisy tweets produces garbage clusters. Manual definition from reading hundreds of tweets + cross-checking with Spotify's help center structure produced clean, actionable categories. |
| 3 | **Used keyword matching to bootstrap labels** | Chicken-and-egg: need labels to train a classifier, need a classifier to label. Keywords break the loop. 75% coverage is rough but enough to train something better. |
| 4 | **Kept TF-IDF classifier as baseline, switched to LLM for production** | TF-IDF (84.5%) is a proper ML baseline with honest test metrics. LLM is better but harder to evaluate rigorously. Having both satisfies the "two baselines" requirement and lets us compare fairly. |
| 5 | **Scraped help articles rather than using tweet resolutions** | Only 4% of SpotifyCares link shares pointed to actual fix articles. 48% were "DM us". The tweets don't contain resolutions — the help center does. |
| 6 | **Replaced TF-IDF retrieval with sentence-transformers + ChromaDB** | TF-IDF failed on paraphrased queries (50% accuracy). Embeddings got 100%. For 18 chunks a vector DB is overkill, but it proves the approach scales and matches the assignment's implied expectations. |
| 7 | **Filtered historical conversations for customer confirmation** | Not every agent response fixes the issue. "Try restarting" followed by "still broken" is a failed resolution. By filtering for "thanks" / "that worked" after the agent's fix, we only keep the 856 proven resolutions out of 26,068 conversations. |
| 8 | **Used two ChromaDB collections, not one** | KB articles (WHAT to say) and historical conversations (HOW to say it) serve different purposes. Mixing them in one collection would dilute retrieval quality. Two collections let us control the balance in the prompt. |
| 9 | **Added `/no_think` to all Groq prompts** | Qwen3 models output chain-of-thought `<think>` blocks by default. This clutters the response. `/no_think` suppresses it. We also strip any residual `<think>` tags in post-processing. |
| 10 | **LLM prompt says "pick ONE diagnostic question"** | Without this rule, the LLM asks 3-4 questions at once, which is overwhelming. Real SpotifyCares agents ask one question per reply. This rule, combined with historical examples, produces natural diagnostic flows. |
| 11 | **Included 3 historical conversations in prompt, not 1 or 10** | 1 is too few — the LLM might over-index on a single pattern. 10 blows up the context window and adds noise. 3 gives enough pattern diversity while keeping the prompt focused. |
| 12 | **Used Groq (free tier) over OpenAI** | Cost. Groq offers free API access with generous rate limits. qwen3.8-27b is capable enough for intent classification and response generation. For production, we'd evaluate GPT-4o or Claude. |
| 13 | **Redirected billing/account queries to human support** | The bot cannot access Spotify accounts. Attempting to handle "I was charged twice" without account access would produce hallucinated or useless responses. Honest redirection is better than a fake resolution. |
| 14 | **Kept diagnostics.json separate from kb.json** | KB articles are retrieved semantically (embedded in ChromaDB). Diagnostic questions are injected directly into every prompt for the matching intent. Different access patterns = different files. |
| 15 | **Curated 25 conversations per intent (max), not all 856** | Many resolved conversations are near-duplicates ("try restarting" → "worked, thanks!" appears hundreds of times). Deduplication by first agent response keeps 223 diverse examples without redundancy. |

---

### 7. Golden Evaluation Set

200 tweets randomly sampled (proportional to intent distribution) from the 26,068 SpotifyCares conversations. Each tweet was hand-labeled with:
- **gold_intent**: the correct intent (not keyword-matched — read and judged individually)
- **difficulty**: easy / medium / hard
- **notes**: reasoning for the label

**Key finding**: The keyword labeler disagreed with gold labels on 97/200 tweets (48%). The largest error source was `device_compatibility` absorbing tweets about account access, billing, playback, and content availability whenever a device name appeared.

**Sampling note**: Proportional random sampling with a minimum floor of 3 per intent. 20 tweets reserved for other/ambiguous. Random seed 42 for reproducibility.

Files: `data/golden_eval_set.json`, `data/golden_eval_meta.json`

---

### 8. Evaluation Harness Results

Ran the full bot pipeline on 50 randomly sampled golden examples (150 Groq API calls across 4 batched phases).

#### Intent Classification (on gold-labeled data)

| Metric | Value |
|---|---|
| Intent accuracy | **84% (42/50)** |
| Easy tweets | 81.4% (n=43) |
| Medium tweets | 100% (n=7) |

#### Escalation Accuracy

| Metric | Value |
|---|---|
| Escalation accuracy | 34% (17/50) |
| Issue | Bot includes "contact support" link too often — even when it can resolve the issue itself |

#### LLM-as-Judge Response Quality (1-5 scale)

| Dimension | Score | Meaning |
|---|---|---|
| Tone | **4.74** | Near-perfect SpotifyCares style |
| Actionability | **3.96** | Clear steps in most responses |
| Factual | **3.80** | Grounded in KB, occasional inaccuracies |
| Diagnostic | **3.72** | Usually asks the right questions |
| **Overall** | **4.05** | |

#### Human-Judge Agreement (Cohen's Kappa)

50 bot responses scored independently by a human and the LLM judge on the same 4-dimension rubric:

| Dimension | Kappa | Exact Match | Within ±1 |
|---|---|---|---|
| Tone | 0.781 | 92% | 100% |
| Actionability | 0.742 | 84% | 98% |
| Diagnostic | 0.655 | 76% | 84% |
| Factual | 0.600 | 72% | 82% |
| **Overall** | **0.709** | **81%** | **91%** |

**Interpretation: Substantial agreement** (Kappa = 0.709). The LLM judge is reliable enough for automated evaluation — it agrees with human scoring 81% exactly and 91% within ±1 point.

Files: `data/eval_results.json`, `data/eval_summary.json`, `data/eval_human_agreement.json`

---

### 9. Escalation Logic

The bot has two types of escalation:

**Instant escalation** — some queries go to a human immediately:
- Hacked/compromised accounts (security)
- Fraud/unauthorized charges
- Critical urgency (profanity, all caps, threats to leave)

**Conversation-aware escalation** — the bot tracks:
- Turn count (max 5 before auto-escalate)
- Failed fix attempts (max 3 before escalate)
- User frustration signals ("still broken", "nothing works", "ridiculous")

**Urgency levels** (printed with every response):
- LOW: normal question
- MEDIUM: mild frustration
- HIGH: clearly angry
- CRITICAL: threats, profanity, all caps → instant escalate

**Escalation includes a handoff summary** for the human agent: issue type, urgency, fixes already attempted, customer messages.

---

## Pipeline Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    USER SENDS MESSAGE                     │
└─────────────────────────┬────────────────────────────────┘
                          │
              ┌───────────┴───────────┐
              │  URGENCY DETECTION    │  Scans for frustration signals
              │  → LOW/MEDIUM/HIGH/   │  (profanity, caps, threats)
              │    CRITICAL           │
              └───────────┬───────────┘
                          │
              ┌───────────┴───────────┐
              │  INTENT CLASSIFICATION│  Groq LLM (qwen3.8-27b)
              │  → 1 of 12 intents   │
              └───────────┬───────────┘
                          │
              ┌───────────┴───────────┐
              │  ESCALATION CHECK     │  Instant: hacked/fraud/critical
              │                       │  Turn-based: 3 failed fixes or
              │  → AUTO or ESCALATE   │  5 turns without resolution
              └─────┬─────────┬───────┘
                    │         │
              AUTO  │         │ ESCALATE
                    ▼         ▼
  ┌─────────────────────┐  ┌──────────────────────────┐
  │ KB + Historical     │  │ Handoff to human agent   │
  │ retrieval (ChromaDB)│  │ with summary of issue,   │
  │         ↓           │  │ fixes tried, urgency     │
  │ LLM response        │  └──────────────────────────┘
  │ generation          │
  └─────────────────────┘
```

## Evolution (4 Iterations)

| Component | V1 (TF-IDF only) | V2 (+ LLM) | V3 (+ Embeddings) | V4 (+ Historical) |
|---|---|---|---|---|
| Intent classifier | TF-IDF + LogReg | Groq LLM | Groq LLM | Groq LLM |
| KB retrieval | TF-IDF | TF-IDF | Embeddings | Embeddings |
| Historical retrieval | None | None | None | Embeddings (223 convos) |
| Response | Template dump | Groq LLM + KB | Groq LLM + KB | Groq LLM + KB + Historical |
| Grounded in tweets? | No | No | No | **Yes** |
| Diagnostic flow? | No | No | No | **Yes** |

---

## File Structure

```
spotify_bot/
├── bot.py                              ← Interactive bot (run this)
├── README.md                           ← This file
├── FLOW.md                             ← Detailed technical documentation
│
├── scripts/
│   ├── 01_extract_conversations.py     ← Extract 26K conversations from CSV
│   ├── 02_label_intents.py             ← Keyword-based intent labeling
│   ├── 03_build_knowledge_base.py      ← Build KB + diagnostics
│   ├── 04_train_intent_classifier.py   ← TF-IDF + LogReg baseline
│   ├── 05_build_rag_pipeline.py        ← Embed KB into ChromaDB
│   ├── 06_test_bot.py                  ← V1 end-to-end tests
│   ├── 07_extract_historical_responses.py  ← Mine resolved conversations
│   ├── 08_embed_historical.py          ← Embed historical into ChromaDB
│   ├── 09_evaluation_harness.py        ← Full evaluation (batched, 4 phases)
│   └── kb_retriever.py                 ← Shared retriever class
│
├── data/                               ← Processed data files
│   ├── golden_eval_set.json            ← 200 hand-labeled golden examples
│   ├── golden_eval_meta.json           ← Sampling + labeling methodology
│   ├── eval_results.json               ← Full evaluation results (50 examples)
│   ├── eval_summary.json               ← Headline evaluation metrics
│   ├── eval_human_agreement.json       ← Cohen's Kappa (human vs LLM judge)
│
└── knowledge_base/                     ← Knowledge + retrieval
    ├── kb.json                         ← 18 structured help articles
    ├── diagnostics.json                ← Per-intent diagnostic flows
    ├── chunks.json                     ← KB chunks (reference)
    ├── historical_responses.json       ← 223 curated resolved conversations
    └── chroma_db/                      ← Vector store (KB + historical)
```

---

## Detailed Documentation

For the full technical deep-dive (1,000+ lines) covering data exploration, brand selection analysis, URL analysis, cross-checks, and all implementation details, see **[FLOW.md](FLOW.md)**.
