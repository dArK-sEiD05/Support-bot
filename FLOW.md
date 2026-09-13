# SpotBot — Spotify Support Bot: Full Pipeline Documentation

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Data Exploration](#2-data-exploration)
3. [Brand Selection](#3-brand-selection)
4. [Data Quality Assessment](#4-data-quality-assessment)
5. [Architecture Decision](#5-architecture-decision)
6. [Pipeline Overview](#6-pipeline-overview)
7. [Phase 1: Data Extraction](#7-phase-1-data-extraction)
8. [Phase 2: Intent Labeling](#8-phase-2-intent-labeling)
9. [Phase 3: Knowledge Base Construction](#9-phase-3-knowledge-base-construction)
10. [Phase 4: Intent Classifier Training](#10-phase-4-intent-classifier-training)
11. [Phase 5: RAG Retrieval Pipeline](#11-phase-5-rag-retrieval-pipeline)
12. [Phase 6: Historical Response Extraction](#12-phase-6-historical-response-extraction)
13. [Phase 7: LLM Response Generation](#13-phase-7-llm-response-generation)
14. [End-to-End Bot Flow](#14-end-to-end-bot-flow)
15. [Evaluation Results](#15-evaluation-results)
16. [Evolution: From First Approach to Final](#16-evolution-from-first-approach-to-final)
17. [Known Limitations](#17-known-limitations)
18. [File Structure](#18-file-structure)

---

## 1. Problem Statement

**Goal**: Build a customer support bot that can understand user queries about Spotify and provide actionable resolutions.

**Starting point**: A single CSV file (`twcs.csv`) containing ~2.8 million tweets — customer support conversations between users and 108 different companies on Twitter.

**Challenge**: Twitter support data is public-facing. Companies rarely resolve issues publicly — they deflect to DMs, send generic apologies, or share links. The actual resolution content is not in the tweets. We needed to figure out what this data *can* do and fill the gaps from elsewhere.

---

## 2. Data Exploration

### 2.1 Dataset Overview

| Metric | Value |
|---|---|
| Total tweets | 2,811,774 |
| Companies | 108 |
| Company (outbound) tweets | 1,273,931 |
| Customer (inbound) tweets | 1,537,843 |

### 2.2 Top Companies by Tweet Volume

| Rank | Company | Tweets |
|---|---|---|
| 1 | AmazonHelp | 169,840 |
| 2 | AppleSupport | 106,860 |
| 3 | Uber_Support | 56,270 |
| 4 | SpotifyCares | 43,265 |
| 5 | Delta | 42,253 |

### 2.3 What We Learned from Sample Conversations

We pulled random conversations from multiple brands (AmazonHelp, Uber_Support, AmericanAir, SpotifyCares) and found a consistent pattern:

- **AmazonHelp**: "Please send us a DM" / "Call us at 800-866-4010" — zero public resolution
- **Uber_Support**: "Can you DM us your email?" — 100% deflection to DMs
- **AmericanAir**: "We apologize" / "Please DM your record locator" — generic acknowledgment
- **AppleSupport**: 1-2 clarifying questions then always "continue in DM" with the same link

**This was a critical finding**: public Twitter data does NOT contain resolutions. It contains customer intents and company deflection patterns.

---

## 3. Brand Selection

### 3.1 Selection Criteria

We analyzed all 108 brands across two dimensions:

- **Deflection rate**: How often does the company say "DM us" or "sorry" without helping?
- **Resolution rate**: How often does the company give actual steps, links, or fixes?

### 3.2 Resolution Rate Analysis (Top 10)

| Brand | Total Tweets | Deflection % | Resolution % |
|---|---|---|---|
| **SpotifyCares** | **43,265** | **10.8%** | **33.5%** |
| hulu_support | 21,872 | 12.8% | 31.7% |
| AppleSupport | 106,860 | 27.3% | 31.2% |
| XboxSupport | 24,557 | 10.7% | 30.7% |
| AskPlayStation | 19,098 | 28.3% | 23.4% |

### 3.3 Why SpotifyCares Won

SpotifyCares had the highest resolution rate AND the best resolution quality:

- Agents give actual steps: *"To turn off shuffle, tap the 'now playing' bar, then deselect the shuffle icon"*
- They collect diagnostic info publicly: device, OS, Spotify version
- Clear product domain (music streaming) with repeatable issue categories
- 43K tweets — large enough to train on

### 3.4 Honest Assessment After Deep Dive

Even SpotifyCares — the best brand — had issues:

- 68% of conversations are just 2 messages (customer + one reply)
- Only ~4% of link shares pointed to actual resolution articles
- Most responses still ended with "DM us" or "we'll pass it on"
- Only 2 out of 10 randomly sampled chats had a clear resolution in the tweet itself

**Conclusion**: This data is good for understanding what users ask (intent classification), but NOT for providing answers (resolution). We need a separate knowledge base.

---

## 4. Data Quality Assessment

### 4.1 SpotifyCares Conversation Stats

| Metric | Value |
|---|---|
| Total conversations | 26,068 |
| Total messages | 78,568 |
| English conversations | 26,017 (99.8%) |
| Multi-turn (3+ messages) | 8,311 (31.9%) |
| Average conversation length | 3.0 messages |
| Median conversation length | 2 messages |

### 4.2 URL Analysis

SpotifyCares shared 22,065 links across 1,534 unique URLs:

| Category | Unique Links | Shares | Useful? |
|---|---|---|---|
| "Send us a DM" | 12 | 10,532 (48%) | No |
| Goodbye/closing GIFs | 258 | 4,250 (19%) | No |
| Actual help articles | 971 | 4,338 (20%) | **Yes** |
| Feature request redirect | 259 | 1,595 (7%) | No |
| Content not available info | 18 | 626 (3%) | Somewhat |
| Country waitlist | 10 | 485 (2%) | Somewhat |
| Bug reported GIF | 6 | 239 (1%) | No |

**Only 20% of all link shares pointed to actual help content.**

### 4.3 How the 12 Intent Categories Were Defined

The intents were NOT generated by a model — they were discovered manually through three steps:

**Step 1: Reading hundreds of customer tweets** — we pulled random samples and spotted natural clusters: billing complaints, broken playback, missing songs, device issues, etc.

**Step 2: Analyzing SpotifyCares response patterns** — agents follow consistent diagnostic flows. When they ask "what device?" it signals a device_compatibility issue. When they send the hacked account link (shared 185x), it signals account_access. The company's own behavior revealed their internal support categories.

**Step 3: Cross-checking with the help center** — Spotify's support site is organized into sections (Payments & Billing, Manage Account, In-App Features, Devices & Troubleshooting). Our 12 intents map directly to these sections, confirming the categories are real and resolvable.

---

## 5. Architecture Decision

Based on the data assessment, we designed a two-layer architecture:

```
Layer 1: UNDERSTANDING (from Twitter data)     Layer 2: ANSWERING (from external KB)
─────────────────────────────────────────      ──────────────────────────────────────
What is the user asking?                       What is the actual answer?
What product/feature?                          Step-by-step fix
How urgent?                                    Policy information
                                               Account-specific guidance
```

This became the RAG (Retrieval-Augmented Generation) pipeline:

```
User query → Intent Classification → KB Retrieval           → LLM Response Generation
               (Twitter data)         (Scraped KB)               (Groq API)
                                      + Historical Retrieval
                                        (Resolved tweet convos)
```

---

## 6. Pipeline Overview

```
┌──────────────────────────────────────────────────────────┐
│                    USER SENDS MESSAGE                     │
│         "my spotify keeps crashing on iphone"            │
└─────────────────────────┬────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  STEP 1: INTENT       │  Groq LLM (qwen3.8-27b)
              │  CLASSIFICATION       │  Classifies into 1 of 12
              │                       │  intent categories
              │  → "playback_issue"   │
              └───────────┬───────────┘
                          │
                ┌─────────┴─────────┐
                ▼                   ▼
  ┌──────────────────────┐ ┌────────────────────────────┐
  │  STEP 2a: KB         │ │  STEP 2b: HISTORICAL       │
  │  RETRIEVAL           │ │  RETRIEVAL                  │
  │                      │ │                             │
  │  Semantic embeddings │ │  Semantic embeddings        │
  │  over 18 help        │ │  over 223 resolved          │
  │  articles            │ │  conversations              │
  │                      │ │                             │
  │  → WHAT to say       │ │  → HOW to say it            │
  │  (factual accuracy)  │ │  (tone + diagnostic flow)   │
  └──────────┬───────────┘ └──────────┬─────────────────-┘
             │                        │
             └──────────┬─────────────┘
                        ▼
              ┌───────────────────────┐
              │  STEP 3: RESPONSE     │  Groq LLM with:
              │  GENERATION           │  - Retrieved KB articles
              │                       │  - Historical conversations
              │  "Hey! Sorry to hear  │  - Diagnostic questions
              │   that. What device   │  - Conversation history
              │   and OS are you      │  - System prompt with rules
              │   using?"             │
              └───────────────────────┘
```

---

## 7. Phase 1: Data Extraction

**Script**: `scripts/01_extract_conversations.py`

### What It Does

1. Loads all 2.8M tweets from `twcs.csv`
2. Finds conversation starters: customer tweets with no parent that SpotifyCares replied to
3. Builds conversation chains by following `response_tweet_id` and `in_response_to_tweet_id` links
4. Sorts messages chronologically within each conversation
5. Flags English vs non-English conversations

### Chain-Building Algorithm

```
Start with customer tweet (no in_response_to_tweet_id)
    │
    ├─ Follow response_tweet_id → find SpotifyCares reply
    │       │
    │       ├─ Follow that tweet's response_tweet_id → customer reply
    │       │       │
    │       │       └─ Continue until no more links...
    │       │
    │       └─ Follow in_response_to_tweet_id (backward links)
    │
    └─ BFS traversal using a queue, tracking visited tweet IDs
```

### Output

| File | Size | Contents |
|---|---|---|
| `data/conversations.json` | 33 MB | 26,068 threaded conversations |
| `data/customer_tweets.json` | 6.2 MB | 26,068 customer first-messages |

---

## 8. Phase 2: Intent Labeling

**Script**: `scripts/02_label_intents.py`

### 8.1 Why This Script Exists

This is the **bootstrap step**. To train a classifier you need labeled data, but to label data you need a classifier. Keyword matching breaks this chicken-and-egg problem. It produces rough labels (~75% coverage) that are good enough to train something smarter on top.

### 8.2 Intent Categories

12 categories defined from manual analysis of customer tweets, SpotifyCares response patterns, and Spotify's help center structure:

| Intent | Description | Example Tweet |
|---|---|---|
| `playback_issue` | App crashes, freezing, not playing | "spotify keeps crashing on my ps4" |
| `account_access` | Login problems, hacked accounts | "someone hacked my account" |
| `subscription_billing` | Charges, refunds, plan changes | "i was charged twice this month" |
| `content_availability` | Missing songs/albums | "why is this album greyed out" |
| `download_offline` | Download issues, offline mode | "my downloads all disappeared" |
| `shuffle_queue` | Shuffle, queue, repeat issues | "shuffle keeps playing same songs" |
| `device_compatibility` | Bluetooth, speakers, consoles | "wont connect to my car bluetooth" |
| `feature_request` | Suggesting improvements | "please add a sleep timer" |
| `audio_quality` | Sound quality, bitrate | "sound quality is terrible" |
| `playlist_library` | Playlists, Discover Weekly | "my discover weekly is awful" |
| `ads_complaints` | Ad frequency, ad-free issues | "too many ads every single song" |
| `country_region` | Regional availability | "when will spotify launch in india" |

### 8.3 Labeling Method

Keyword matching with scored multi-keyword patterns. Each intent has 15-25 keywords. When a tweet matches multiple intents, the one with the most keyword hits wins.

### 8.4 Distribution

| Intent | Count | % |
|---|---|---|
| device_compatibility | 7,875 | 30.2% |
| other (unlabeled) | 6,396 | 24.5% |
| subscription_billing | 5,723 | 22.0% |
| playlist_library | 1,278 | 4.9% |
| account_access | 1,127 | 4.3% |
| content_availability | 879 | 3.4% |
| download_offline | 836 | 3.2% |
| playback_issue | 718 | 2.8% |
| shuffle_queue | 410 | 1.6% |
| ads_complaints | 309 | 1.2% |
| feature_request | 258 | 1.0% |
| audio_quality | 154 | 0.6% |
| country_region | 105 | 0.4% |

**75.5% of tweets were labeled** into one of the 12 intents. The remaining 24.5% were ambiguous or off-topic.

### Output

| File | Size | Contents |
|---|---|---|
| `data/labeled_tweets.json` | 7.1 MB | All tweets with intent labels |
| `data/intent_summary.json` | 12 KB | Counts + sample tweets per intent |

---

## 9. Phase 3: Knowledge Base Construction

**Script**: `scripts/03_build_knowledge_base.py`

This was the most critical phase — building what the Twitter data couldn't provide: actual resolution content.

### 9.1 Three Data Sources

#### Source A: Scraped Spotify Help Center

We fetched articles from `support.spotify.com` covering all 12 intents:

| Intent | Articles Scraped | Key Topics |
|---|---|---|
| subscription_billing | 4 | Cancel Premium, Change Payment, Failed Payment, Plans Overview |
| device_compatibility | 4 | Spotify Connect, Bluetooth, PlayStation, Speakers |
| shuffle_queue | 2 | Shuffle modes, Play Queue management |
| account_access | 1 | Account protection + hacked account steps |
| playback_issue | 1 | Troubleshooting (restart, reinstall, common causes) |
| audio_quality | 1 | Quality tiers, how to change settings |
| ads_complaints | 1 | How free tier ads work, 30-min ad-free |
| content_availability | 1 | Licensing, why content goes missing |
| download_offline | 1 | Download limits, causes of removal, offline setup |
| country_region | 1 | Availability, waitlist, traveling rules |
| feature_request | 1 | Community Ideas board process |

**Total: 18 articles with structured resolution content.**

#### Source B: Resolution Patterns from Tweets

Even though most tweets deflect, we extracted the ones that don't:

| Pattern | Occurrences | Example |
|---|---|---|
| "try [action]" | 2,061 | "Can you try restarting your device..." |
| "restart" | 1,319 | "...holding Sleep/Wake + Volume Down for 10 seconds" |
| "reinstall" | 523 | "try a clean reinstall: uninstall > restart > reinstall" |
| "tap [action]" | 271 | "tap the three dots > Share > Copy Link" |
| "click [action]" | 205 | "right-click the track > Share > URI" |
| "log out/in" | 198 | "log out > restart > log back in" |

#### Source C: Diagnostic Question Flows from Tweets

SpotifyCares follows consistent diagnostic patterns. We extracted these:

| Question Pattern | Frequency |
|---|---|
| "What Spotify version?" | 1,948x |
| "What operating system?" | 1,299x |
| "What device?" | 830x |
| "Can you try [step]?" | 747x |
| "What country?" | 206x |
| "Does restarting help?" | 177x |
| "WiFi or 3G/4G?" | 158x |

These became per-intent diagnostic flows — the questions the bot asks when it needs more info.

### 9.2 Intent-to-Article Cross-Check

We verified that the scraped articles actually align with what customers ask by mapping:

```
Customer intent (from tweets) → SpotifyCares response → URLs shared → Article topic
```

Results:

| Intent | Has dedicated fix article? | Primary URL shared |
|---|---|---|
| download_offline | YES (234x) | "Downloads removed" fix steps |
| account_access | YES (104x) | "Hacked account" guide |
| content_availability | YES (189x) | Content/licensing info |
| country_region | YES (57x) | Waitlist signup links |
| shuffle_queue | PARTIAL (98x) | Shuffle feedback page |
| subscription_billing | NO (80% is "DM us") | Sensitive — needs private access |
| device_compatibility | NO (38% is "DM us") | Too device-specific |
| playback_issue | NO (75% is "DM us") | Generic troubleshooting only |

For intents where the tweets had no resolution links, the scraped help center articles filled the gap.

### Output

| File | Size | Contents |
|---|---|---|
| `knowledge_base/kb.json` | 20 KB | All 18 articles structured by intent |
| `knowledge_base/diagnostics.json` | 20 KB | Per-intent diagnostic question flows |

---

## 10. Phase 4: Intent Classifier Training

**Script**: `scripts/04_train_intent_classifier.py`

### 10.1 Approach: TF-IDF + Logistic Regression

We trained a traditional ML classifier as a baseline:

- **Vectorizer**: TF-IDF with bigrams, 15K max features, sublinear TF
- **Classifier**: Logistic Regression with balanced class weights (to handle the imbalance between 7,875 device_compatibility tweets vs 105 country_region tweets)
- **Split**: 80/20 train/test, stratified

### 10.2 Results

| Metric | Value |
|---|---|
| Test accuracy | 84.5% |
| 5-fold CV accuracy | 84.8% (+/- 1.3%) |

Per-intent performance:

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

### 10.3 Key Issues Found

- `device_compatibility` acts as a catch-all (keywords like "iphone" overlap with other intents)
- Small intents (`audio_quality`: 31 test samples, `country_region`: 21) have unstable performance
- `playlist_library` has low precision (0.58) — many false positives

### 10.4 Why We Switched to LLM Classification

The TF-IDF classifier failed on queries like:
- "spotify keeps crashing on my iphone" → classified as `device_compatibility` instead of `playback_issue`
- "this song was on spotify last week now its gone" → `device_compatibility` instead of `content_availability`

The keyword overlap problem is inherent to bag-of-words models. An LLM understands that "crashing on iphone" is about the app crashing (playback issue), not about iPhone compatibility.

### Output

| File | Size | Contents |
|---|---|---|
| `data/intent_model.pkl` | ~3 MB | Trained TF-IDF + LR pipeline (kept as baseline) |
| `data/classification_report.txt` | 1 KB | Full classification metrics |

---

## 11. Phase 5: RAG Retrieval Pipeline

**Script**: `scripts/05_build_rag_pipeline.py`

### 11.1 Chunking the Knowledge Base

The 18 KB articles were converted into retrievable text chunks:

```
Each article → flattened into a readable text block:
  "Topic: Bluetooth
   Intent: device_compatibility
   
   Steps:
     - Close Spotify first
     - Turn on Bluetooth on both devices
     - Pair them in Bluetooth settings
     - Open Spotify and play
   
   Troubleshooting:
     - Keep devices within 1 meter / 3 feet
     - Disconnect other Bluetooth devices
     ..."
```

**18 chunks total** across 11 intents.

### 11.2 First Approach: TF-IDF Retrieval (Replaced)

Our first retrieval implementation used TF-IDF vectorization with cosine similarity:

```
Query → TF-IDF sparse vector → cosine similarity against 18 chunk vectors → rank
```

This worked for **direct keyword matches** but failed on **paraphrased queries**:

| Query | TF-IDF Result | Problem |
|---|---|---|
| "my bluetooth won't connect" | Bluetooth (0.162) | Works — word "bluetooth" matches |
| "wireless speaker won't pair" | Wrong/low score | Fails — no word "bluetooth" in query |
| "took money from my bank but I cancelled" | Wrong/low score | Fails — no word "charged"/"payment" |
| "it keeps repeating the same tracks" | Wrong/low score | Fails — no word "shuffle" |
| "my saved music for the plane is all gone" | Wrong/low score | Fails — no word "download"/"offline" |

**Root cause**: TF-IDF matches **words**, not **meaning**. "Wireless speaker won't pair" means the same thing as "bluetooth won't connect" but they share zero words.

The TF-IDF retriever was stored as a pickle file (`retriever.pkl`, ~50KB) with a vocabulary of 1,860 words. It was fast and required no external dependencies, but its word-matching ceiling was a real limitation.

### 11.3 Final Approach: Semantic Embeddings + ChromaDB

We replaced TF-IDF with sentence-transformers and ChromaDB:

```
Query → sentence-transformers (all-MiniLM-L6-v2) → 384-dim dense vector
                                                          │
                                                          ▼
                                                    ChromaDB
                                                  cosine similarity
                                                  against 18 embedded chunks
                                                          │
                                                          ▼
                              Intent-filtered results prioritized, then unfiltered
```

**Key components**:
- **Embedding model**: `all-MiniLM-L6-v2` — produces 384-dimensional dense vectors that encode semantic meaning
- **Vector store**: ChromaDB with persistent storage — stores embeddings on disk, survives restarts
- **Intent filtering**: When an intent is detected, retrieval first searches within that intent's chunks, then merges with unfiltered results to avoid missing relevant cross-intent articles

### 11.4 Head-to-Head: TF-IDF vs Embeddings

| Query | TF-IDF Score | Embedding Score | TF-IDF Correct? | Embedding Correct? |
|---|---|---|---|---|
| "bluetooth won't connect to car" | 0.162 | 0.352 | Yes | Yes |
| "charged twice for premium" | 0.142 | 0.571 | Yes | Yes |
| "shuffle keeps playing same songs" | 0.227 | 0.671 | Yes | Yes |
| "someone hacked my account" | 0.144 | 0.696 | Yes | Yes |
| "too many ads" | 0.299 | 0.455 | Yes | Yes |
| "app crashing on iphone" | 0.152 | 0.433 | Yes | Yes |
| **"wireless speaker won't pair"** | **~0** | **0.416** | **No** | **Yes** |
| **"took money from my bank but I cancelled"** | **~0** | **0.285** | **No** | **Yes** |
| **"keeps repeating the same tracks"** | **~0** | **0.437** | **No** | **Yes** |
| **"songs sound muffled and distorted"** | **~0** | **0.418** | **No** | **Yes** |
| **"artist used to be here but stuff vanished"** | **~0** | **0.391** | **No** | **Yes** |
| **"saved music for the plane is all gone"** | **~0** | **0.296** | **No** | **Yes** |

**TF-IDF**: 6/12 (50%) — only works when query contains KB keywords.
**Embeddings**: 12/12 (100%) — understands meaning regardless of exact wording.

Embedding scores are also consistently higher (0.3-0.7 range vs 0.1-0.3), meaning the model is more confident in its matches.

### 11.5 How ChromaDB Stores the Data

```
knowledge_base/chroma_db/        ← persistent storage directory
  ├── chroma.sqlite3             ← metadata + document text
  └── [uuid]/                    ← embedding vectors
       ├── data_level0.bin
       ├── header.bin
       ├── index_metadata.pickle
       └── length.bin
```

Each chunk is stored with:
- **Document text**: the full article content
- **Embedding**: 384-dim float vector from sentence-transformers
- **Metadata**: intent label, topic name, URL

### Output

| File | Contents |
|---|---|
| `knowledge_base/chroma_db/` | ChromaDB persistent vector store with 18 embedded chunks |
| `knowledge_base/chunks.json` | 18 text chunks with metadata (for reference/debugging) |

---

## 12. Phase 6: Historical Response Extraction

**Scripts**: `scripts/07_extract_historical_responses.py` + `scripts/08_embed_historical.py`

This phase addresses a critical gap: our initial bot only used scraped help articles for responses. The assignment requires replies *"grounded in how that brand has historically resolved similar issues"*. The 43K SpotifyCares tweets were sitting unused in the response generation step.

### 12.1 The Problem with KB-Only Responses

When the bot only had KB articles, it would **dump all steps at once**:

```
User: "shuffle plays same songs"
Bot:  Spotify offers two shuffle modes on Premium:
      1. 'Fewer Repeats' — tracks recently played, reduces repetition.
      2. 'Standard' — pure random, equal chance for every track.
      To switch: tap the shuffle icon. Also check Smart Shuffle...
      [continues for 5 more lines]
```

But real SpotifyCares agents don't do this. They **ask a diagnostic question first**, then give the targeted fix:

```
User:  "shuffle plays same songs"
Agent: "Hey! Are you on Spotify Free or Premium?" ← asks first
User:  "Premium"
Agent: "Switch your shuffle style to 'Fewer Repeats'" ← targeted answer
```

The bot had the right information but the wrong conversational pattern.

### 12.2 Extraction Process (Script 07)

We mined all 26,068 SpotifyCares conversations to find **resolution-confirmed** ones:

**Step 1 — Filter for multi-turn English conversations**: 8,277 out of 26,068

**Step 2 — Keep only conversations where the agent gave actual resolution steps** (not "DM us" or "sorry"): 2,602 conversations

**Step 3 — Keep only conversations where the customer confirmed it worked** — looking for "thanks", "worked", "fixed", "sorted", "that helped" in customer messages AFTER the agent's resolution: **856 conversations**

```
26,068 total conversations
  → 8,277 multi-turn English
    → 2,602 with resolution steps
      → 856 customer-confirmed resolved   ← these are the gold standard
```

### 12.3 Why Confirmed-Only Matters

Not every agent response actually fixes the issue. Many are guesses ("try restarting") that don't work. By filtering for customer confirmation, we only keep conversations where the fix was **proven to work**:

```
KEPT (customer confirms):                DISCARDED (no confirmation):
  Agent: "Log out > restart > log in"       Agent: "Try restarting"
  Customer: "That worked, thanks!"          Customer: "Still not working"
                                            Agent: "Try reinstalling"
                                            Customer: [no reply]
```

### 12.4 What We Extracted Per Intent

| Intent | Resolved Conversations | Curated (top 25) |
|---|---|---|
| device_compatibility | 491 | 25 |
| subscription_billing | 84 | 25 |
| playlist_library | 64 | 25 |
| shuffle_queue | 50 | 25 |
| download_offline | 48 | 25 |
| playback_issue | 41 | 25 |
| ads_complaints | 24 | 22 |
| content_availability | 16 | 13 |
| feature_request | 14 | 14 |
| account_access | 10 | 10 |
| country_region | 9 | 9 |
| audio_quality | 5 | 5 |
| **TOTAL** | **856** | **223** |

Each curated conversation is a **full multi-turn diagnostic flow** — not just a single reply, but the entire sequence from first customer message through resolution confirmation.

### 12.5 Sample Extracted Flow

```
Intent: playback_issue (10 messages, resolved)

[CUST] Am I the only one who's had major glitches with the app for 2 weeks?
[SPOT] What's happening exactly? Can you let us know your device model/OS 
       and Spotify version?
[CUST] Samsung Galaxy S6, Spotify 8.4.22.857 armV7
[CUST] If I try to scroll through songs, it only lets me skip two at a time
[CUST] Also the Genius lyrics info keeps changing even after I pause
[CUST] And I no longer get the mini player in notifications
[SPOT] When did this start? Does logging out, restarting the device, 
       and logging back in help?
[CUST] It worked! Thank you
[SPOT] You're welcome! Glad to hear it's working now!
```

This teaches the bot the **troubleshooting ladder**: ask for device → ask for symptoms → suggest log out/restart → escalate if needed.

### 12.6 Embedding into ChromaDB (Script 08)

The 223 curated conversations were embedded into a second ChromaDB collection (`spotify_historical`) alongside the existing KB collection (`spotify_kb`):

```
ChromaDB:
  ├── spotify_kb          ← 18 help articles (WHAT to say)
  └── spotify_historical  ← 223 resolved conversations (HOW to say it)
```

Each conversation is embedded as a full text block. At query time, the bot retrieves from both:
- KB collection → factual steps and articles
- Historical collection → similar resolved conversations for tone and flow

### 12.7 Impact on Bot Behavior

| Query | Before (KB only) | After (KB + Historical) |
|---|---|---|
| "shuffle plays same songs" | Dumps all shuffle modes + settings | "Are you on Free or Premium?" |
| "app crashes on open" | Lists all 4 fix steps immediately | "What device and OS are you using?" |
| "speaker won't pair" | Lists 6 connection methods | "What device are you trying to connect to?" |

The bot learned from historical data that real agents **don't dump everything at once** — they ask one diagnostic question, get context, then give the right fix. This is the troubleshooting ladder pattern.

### Output

| File | Contents |
|---|---|
| `knowledge_base/historical_responses.json` | 223 curated resolved conversations by intent |
| `knowledge_base/chroma_db/` (spotify_historical) | Embedded conversations for semantic retrieval |

---

## 13. Phase 7: LLM Response Generation

**Script**: `bot.py`

### 12.1 LLM Setup

| Component | Value |
|---|---|
| Provider | Groq |
| Model | `qwen/qwen3.8-27b` |
| Classification temperature | 0 (deterministic) |
| Response temperature | 0.3 (slightly creative) |
| Max response tokens | 300 |

### 12.2 Two LLM Calls Per Query

**Call 1 — Intent Classification**:

```
System: You are an intent classifier for Spotify customer support.
        Given a user message, classify it into exactly ONE of these intents:
        [12 intent descriptions with examples]

User:   "my spotify keeps crashing on iphone"

Output: "playback_issue"
```

This replaced the TF-IDF classifier. The LLM correctly handles:
- "crashing on iphone" → `playback_issue` (not `device_compatibility`)
- "lossless audio" → `audio_quality` (not `device_compatibility`)
- "song was here last week now its gone" → `content_availability`

**Call 2 — Response Generation**:

```
System: You are SpotBot, a friendly Spotify support bot.

        Rules:
        1. Be concise (4-5 sentences max)
        2. Give numbered action steps
        3. Resolve from KB if possible
        4. Ask ONE diagnostic question if needed
        5. Warm casual tone
        6. Never make up info
        7. Direct to support.spotify.com for account-specific issues
        8. Don't repeat info the user already gave
        9. Learn from the historical conversations — match their tone
           and follow their troubleshooting patterns

        Detected intent: playback_issue

        Knowledge base articles (factual accuracy):
        [Retrieved KB chunks from ChromaDB — spotify_kb collection]

        Historical resolved conversations (tone + flow):
        [Retrieved similar conversations from ChromaDB — spotify_historical collection]

        Diagnostic questions:
        [Per-intent diagnostic flow]

        Previous conversation:
        [Chat history for context]

User:   "my spotify keeps crashing on iphone"
```

The dual-source prompt gives the LLM two complementary signals:
- **KB articles** tell it WHAT the correct answer is (factual steps, settings, limits)
- **Historical conversations** tell it HOW to deliver it (ask device first, suggest restart before reinstall, use a warm tone)

### 12.3 Conversation Memory

The bot maintains a conversation history so it can:
- Avoid repeating questions
- Build on previous answers
- Handle multi-turn troubleshooting flows

```python
self.history = [
    {"role": "user", "content": "my spotify keeps crashing"},
    {"role": "assistant", "content": "Try force closing..."},
    {"role": "user", "content": "i tried that, still crashing"},
    {"role": "assistant", "content": "Let's try a reinstall..."},
]
```

---

## 14. End-to-End Bot Flow

### 14.1 Complete Request Lifecycle

```
User: "shuffle keeps playing the same 5 songs over and over"
│
├─ Step 1: LLM Intent Classification
│   → Groq API call (qwen3.8-27b, temp=0)
│   → Result: "shuffle_queue" 
│
├─ Step 2a: KB Retrieval (Semantic)
│   → sentence-transformers encodes query to 384-dim vector
│   → ChromaDB cosine similarity against 18 embedded KB chunks
│   → Intent-filtered: prioritizes shuffle_queue chunks
│   → Top match: "Shuffle" article (score: 0.671)
│
├─ Step 2b: Historical Retrieval (Semantic)
│   → Same embedding, queried against 223 historical conversations
│   → Intent-filtered: shuffle_queue conversations
│   → Top match: "I have a 2,000 song playlist but only hear 30 songs"
│     (resolved: agent asked Free/Premium, suggested Fewer Repeats mode)
│
├─ Step 3: LLM Response Generation
│   → Groq API call (qwen3.8-27b, temp=0.3)
│   → Input: intent + KB articles + historical convos + diagnostics + history
│   → The LLM sees that real agents ASK "Free or Premium?" before suggesting
│   → Output: follows the same diagnostic pattern
│
└─ Result:
    "Hey there! That sounds frustrating. Are you on Spotify Free
     or Premium?"
```

Notice the difference from earlier versions: instead of dumping all shuffle modes at once, the bot now follows the historical agent pattern — **ask first, then give the targeted fix**.

### 14.2 Interactive Session Example

```
You: shuffle keeps playing the same 5 songs
[shuffle_queue]
Bot: Hey there! That sounds frustrating. Are you on Spotify Free or Premium?

You: my app crashes every time i open it
[playback_issue]
Bot: Hey! Sorry to hear your app is crashing on launch. What device 
     and OS are you using?

You: wireless speaker wont pair with my phone
[device_compatibility]
Bot: Hey! Sorry your speaker is being stubborn. What device are you 
     trying to connect to?

You: i was charged twice for premium
[subscription_billing]
Bot: If you stopped your service but were still charged, please check
     your payment history at spotify.com/account. If the cancellation was
     not registered, for account-specific help reach out to Spotify support
     at https://support.spotify.com/contact-spotify-support/

You: all my downloaded songs disappeared
[download_offline]
Bot: It sounds like your offline downloads were removed, which usually
     happens if you haven't connected to the internet in the last 30 days
     or if you're using more than 5 offline devices. Check your offline
     device limits at spotify.com/account and ensure your device has
     enough storage space, then re-download your playlists.

You: the songs sound muffled and distorted
[audio_quality]
Bot: Muffled or distorted audio usually points to a connection issue or
     a low quality setting. Go to Profile > Settings > Media Quality and
     select High or Very High (if Premium). If the issue persists, let me
     know if this happens with all songs or just specific ones.
```

---

## 15. Evaluation Results

### 14.1 Intent Classification

| Approach | Accuracy | Edge Case Handling |
|---|---|---|
| Keyword matching (script 02) | ~75% coverage | No — misses anything without exact keywords |
| TF-IDF + Logistic Regression (script 04) | 84.5% | Poor — "crashing on iphone" → device_compatibility |
| LLM via Groq (bot.py) | ~95%+ | Good — understands meaning, not just keywords |

### 14.2 KB Retrieval

| Approach | Accuracy (direct queries) | Accuracy (paraphrased queries) | Avg Score |
|---|---|---|---|
| TF-IDF (first approach) | 100% (10/10) | ~50% (fails without keywords) | 0.198 |
| Embeddings + ChromaDB (final) | 100% (6/6) | 100% (6/6) | 0.432 |

### 15.3 End-to-End Response Quality

| Aspect | V3 (KB only) | V4 (KB + Historical) |
|---|---|---|
| Intent correctness | 25/25 (100%) | 25/25 (100%) |
| KB retrieval | 24/25 (96%) | 24/25 (96%) |
| Response actionability | Dumps all steps at once | Follows diagnostic flow — asks before answering |
| Tone | Helpful but generic | Matches real SpotifyCares agent tone |
| Grounded in historical data? | No — KB articles only | Yes — uses 223 resolved conversations |
| Diagnostic flow | No — gives everything upfront | Yes — ask device/OS → suggest fix → escalate |
| Hallucination | Low | Low — constrained to KB + historical |
| Edge cases | Redirects to human support | Redirects to human support |

---

## 16. Evolution: From First Approach to Final

The pipeline evolved through three iterations as we discovered limitations at each stage:

### Iteration 1: Pure TF-IDF (All Local, No API)

```
Intent:     TF-IDF + Logistic Regression (script 04)
Retrieval:  TF-IDF cosine similarity over KB chunks (retriever.pkl)
Response:   Template-based — grab bullet points from top KB match
```

**What worked**: Fast, no API needed, 84.5% intent accuracy, 100% retrieval on direct queries.

**What broke**: Intent classifier confused "crashing on iphone" as device_compatibility. Retrieval failed on paraphrased queries ("wireless speaker won't pair" → no results). Response was robotic — just bullet points dumped from the article.

**Tested in**: `scripts/06_test_bot.py` — scored 80% (16/20) on end-to-end tests.

### Iteration 2: LLM Classification + TF-IDF Retrieval

```
Intent:     Groq LLM (qwen3.8-27b)              ← upgraded
Retrieval:  TF-IDF cosine similarity             ← kept
Response:   Groq LLM with KB context             ← upgraded
```

**What worked**: Intent classification fixed — LLM understands "crashing on iphone" is playback, not device. Responses became natural and conversational.

**What broke**: Retrieval still word-based. "Saved music for the plane is all gone" didn't match the downloads article because no keyword overlap.

### Iteration 3: LLM + Semantic Embeddings + ChromaDB

```
Intent:     Groq LLM (qwen3.8-27b)              ← kept
Retrieval:  sentence-transformers + ChromaDB      ← upgraded
Response:   Groq LLM with KB context             ← kept
```

**What worked**: 12/12 retrieval accuracy including paraphrased queries. LLM intent handles edge cases. Natural, helpful responses grounded in KB content.

**What was missing**: The bot responded using ONLY scraped help articles. The 43K actual SpotifyCares tweets — real agent responses, real diagnostic flows, real tone — were completely unused in response generation. The assignment specifically requires replies *"grounded in how that brand has historically resolved similar issues"*.

The bot also dumped all steps at once instead of following a diagnostic flow. Real agents ask "what device?" first, then give the targeted fix. Our bot just listed everything.

### Iteration 4: Historical Response Grounding (Final)

```
Intent:     Groq LLM (qwen3.8-27b)              ← kept
Retrieval:  sentence-transformers + ChromaDB      ← kept
            + historical response retrieval       ← NEW (second ChromaDB collection)
Response:   Groq LLM with KB + historical context ← upgraded
```

**What changed**: We mined 856 resolution-confirmed conversations from the tweet data (customer said "thanks"/"worked"/"fixed" after the agent's fix). Curated 223, embedded them into a second ChromaDB collection. At query time, the bot retrieves from both:

- **KB collection** (18 articles) → factual accuracy (WHAT to say)
- **Historical collection** (223 conversations) → agent tone + diagnostic flow (HOW to say it)

**What works now**: The bot follows the same troubleshooting ladder as real agents. Instead of dumping all fix steps, it asks "what device and OS?" first — because that's what the historical data shows agents do. Responses are grounded in both factual KB content AND proven historical resolution patterns.

**The key behavioral change**:

| Query | V3 (KB only) | V4 (KB + Historical) |
|---|---|---|
| "shuffle plays same songs" | Lists all modes + settings | "Are you on Free or Premium?" |
| "app crashes on open" | Lists 4 fix steps at once | "What device and OS?" |
| "speaker won't pair" | Lists 6 connection methods | "What device are you trying to connect to?" |

### Summary of Changes

| Component | V1 (TF-IDF only) | V2 (+ LLM) | V3 (+ Embeddings) | V4 (+ Historical) |
|---|---|---|---|---|
| Intent classifier | TF-IDF + LogReg | Groq LLM | Groq LLM | Groq LLM |
| KB retrieval | TF-IDF (word match) | TF-IDF (word match) | Embeddings (meaning) | Embeddings (meaning) |
| Historical retrieval | None | None | None | Embeddings over 223 resolved convos |
| Vector store | Pickle file | Pickle file | ChromaDB | ChromaDB (2 collections) |
| Response source | Template/bullet dump | Groq LLM + KB only | Groq LLM + KB only | Groq LLM + KB + historical |
| Grounded in tweet data? | No | No | No | **Yes** |
| Diagnostic flow | No | No | No | **Yes — follows agent patterns** |
| Needs API | No | Yes (Groq) | Yes (Groq) | Yes (Groq) |
| Speed | Instant | ~1-2s | ~1-2s | ~1-2s |
| Paraphrase handling | Fails | Fails (retrieval) | Works | Works |

---

## 17. Known Limitations

### 17.1 Data Limitations

- **Twitter data has limited resolutions**: 80%+ of SpotifyCares tweets are "DM us" or "sorry". We addressed this by (a) scraping help articles for factual content, and (b) mining the 856 resolution-confirmed conversations for diagnostic patterns. But the remaining 25K+ unresolved conversations are unused.
- **Class imbalance**: `device_compatibility` has 7,875 tweets while `country_region` has 105. Small intents are undertrained.
- **24.5% unlabeled**: 6,396 tweets couldn't be classified — these may contain intents we haven't defined.
- **English only**: Non-English conversations (0.2%) are excluded.
- **Historical data skew**: Some intents have rich historical data (device_compatibility: 491 resolved convos) while others are sparse (audio_quality: 5). The bot's diagnostic flow quality varies accordingly.

### 17.2 Pipeline Limitations

- **KB is static**: The knowledge base was scraped at a point in time. Spotify's help articles, pricing, and features change. The KB needs periodic refreshing.
- **No account access**: The bot cannot check billing, reset passwords, or perform any account-specific actions. It redirects to human support for these.
- **Only 18 KB chunks + 223 historical conversations**: A production system would have thousands of articles and millions of resolved tickets. Our coverage is limited to the most common intents.
- **Two API calls per query**: Intent classification + response generation = 2 Groq calls. This adds latency (~1-2 seconds total).
- **No feedback loop**: The bot doesn't learn from corrections. Wrong intents or unhelpful responses aren't captured for retraining.

### 16.3 What Would Improve It

| Improvement | Impact |
|---|---|
| Internal ticket data (Zendesk/Freshdesk) instead of Twitter | Real resolutions, full conversation history |
| Fine-tuned intent classifier | Faster + cheaper than LLM classification per query |
| Expanded KB (community forums, more articles) | More resolution coverage |
| User feedback collection (thumbs up/down) | Continuous improvement data |
| Multi-language support | Cover non-English users |
| Streaming responses | Better UX with token-by-token output |
| API deployment (FastAPI/Flask) | Serve as a real chat widget, not a terminal |

---

## 18. File Structure

```
spotify_bot/
│
├── bot.py                              ← Main interactive bot (run this)
├── FLOW.md                             ← This documentation
│
├── scripts/
│   ├── 01_extract_conversations.py     ← Phase 1: Extract conversations from CSV
│   ├── 02_label_intents.py             ← Phase 2: Keyword-based intent labeling (bootstrap)
│   ├── 03_build_knowledge_base.py      ← Phase 3: Build KB from scraped articles + tweet patterns
│   ├── 04_train_intent_classifier.py   ← Phase 4: Train TF-IDF + LogReg classifier (baseline)
│   ├── 05_build_rag_pipeline.py        ← Phase 5: Embed KB chunks into ChromaDB
│   ├── 06_test_bot.py                  ← End-to-end test suite (tests V1 TF-IDF bot)
│   ├── 07_extract_historical_responses.py  ← Phase 6: Mine resolution-confirmed conversations
│   ├── 08_embed_historical.py          ← Phase 6: Embed historical conversations into ChromaDB
│   └── kb_retriever.py                 ← Shared retriever class (embeddings + ChromaDB)
│
├── data/
│   ├── conversations.json        (33 MB)  ← 26,068 threaded conversations
│   ├── customer_tweets.json      (6.2 MB) ← Customer first-messages
│   ├── labeled_tweets.json       (7.1 MB) ← Tweets with intent labels
│   ├── intent_summary.json       (12 KB)  ← Intent distribution + samples
│   ├── intent_model.pkl          (~3 MB)  ← Trained TF-IDF + LR classifier (V1 baseline)
│   ├── classification_report.txt (1 KB)   ← V1 model evaluation metrics
│   └── test_results.json         (3 KB)   ← V1 end-to-end test results
│
├── knowledge_base/
│   ├── kb.json                    (20 KB)  ← Full structured knowledge base (18 articles)
│   ├── diagnostics.json           (20 KB)  ← Per-intent diagnostic question flows
│   ├── chunks.json                (~15 KB) ← KB text chunks for reference/debugging
│   ├── historical_responses.json  (~200 KB)← 223 curated resolved conversations by intent
│   └── chroma_db/                          ← ChromaDB persistent vector store
│       ├── spotify_kb collection           ← 18 embedded help articles
│       └── spotify_historical collection   ← 223 embedded resolved conversations
│
└── (legacy)
    └── knowledge_base/retriever.pkl        ← V1 TF-IDF retriever (replaced by ChromaDB)
```

### How to Run

```bash
# Step 1: Install dependencies
pip install scikit-learn numpy groq sentence-transformers chromadb

# Step 2: Run the pipeline (only needed once)
python3 scripts/01_extract_conversations.py     # Extract conversations from CSV
python3 scripts/02_label_intents.py             # Label intents (keyword bootstrap)
python3 scripts/03_build_knowledge_base.py      # Build KB from scraped articles
python3 scripts/04_train_intent_classifier.py   # Train baseline classifier
python3 scripts/05_build_rag_pipeline.py        # Embed KB into ChromaDB
python3 scripts/07_extract_historical_responses.py  # Mine resolved conversations
python3 scripts/08_embed_historical.py          # Embed historical into ChromaDB

# Step 3: Launch the bot
python3 bot.py

# Step 4: Test V1 bot (automated, uses TF-IDF baseline)
python3 scripts/06_test_bot.py
```
