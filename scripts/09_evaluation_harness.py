"""
Script 9: Evaluation Harness (Groq, Batched by Phase)

Phase 1: Classify all 200 intents          → save
Phase 2: Retrieve KB + historical           → save (local, 0 API calls)
Phase 3: Generate all 200 responses         → save
Phase 4: Judge all 200 responses            → save

Each phase resumes from where it left off.
Uses only Groq API with conservative rate limiting.

"""

import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
from groq import Groq
from kb_retriever import KBRetriever

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(BASE_DIR, 'data')
KB_DIR = os.path.join(BASE_DIR, 'knowledge_base')

CLIENT = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
MODEL = "qwen/qwen3.8-27b"
SAMPLE_SIZE = 50  # evaluate on 50 examples (150 API calls total)

INTENTS = [
    "playback_issue", "account_access", "subscription_billing",
    "content_availability", "download_offline", "shuffle_queue",
    "device_compatibility", "feature_request", "audio_quality",
    "playlist_library", "ads_complaints", "country_region", "other",
]

INTENT_DESCRIPTIONS = """
- playback_issue: App crashes, freezing, music stops, buffering, not playing, no sound
- account_access: Can't log in, forgot password, hacked account, locked out
- subscription_billing: Charges, refunds, cancel premium, change plan, payment failed, pricing
- content_availability: Songs/albums missing, greyed out, removed, not available
- download_offline: Downloads disappeared, can't download, offline mode, storage
- shuffle_queue: Shuffle not random, same songs repeating, queue management, autoplay
- device_compatibility: Bluetooth, speakers, car, PS4/PS5, Xbox, Alexa, Chromecast, smart TV
- feature_request: Suggesting new features, requesting improvements
- audio_quality: Sound quality bad, bitrate, lossless, equalizer, distorted, volume
- playlist_library: Discover Weekly, Daily Mix, playlist issues, library, saved songs
- ads_complaints: Too many ads, ad-free not working, annoying ads
- country_region: Spotify not available in country, region restrictions
- other: Greetings, off-topic, unclear intent
"""

SHOULD_ESCALATE = {
    'account_access': ['hacked', 'stolen', 'unauthorized', 'someone else', 'fraud',
                       'changed my email', 'changed my password', 'someone entered',
                       'someone is using', 'someone made'],
    'subscription_billing': ['refund', 'charged twice', 'double charged', 'stole',
                             'charged but', 'money back', 'billed twice', 'unknowingly'],
}


def strip_think(text):
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def call(retries=5, **kwargs):
    for attempt in range(retries):
        try:
            return CLIENT.chat.completions.create(model=MODEL, **kwargs)
        except Exception as e:
            wait = min(15 * (attempt + 1), 90)
            print(f"    [retry {attempt+1}, wait {wait}s: {str(e)[:40]}]", flush=True)
            time.sleep(wait)
    return None


def save(data, name):
    with open(os.path.join(DATA_DIR, name), 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load(name):
    path = os.path.join(DATA_DIR, name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def gold_escalate(intent, text):
    t = text.lower()
    if intent in SHOULD_ESCALATE:
        return any(k in t for k in SHOULD_ESCALATE[intent])
    return intent == 'other'


def run_phase(phase_name, file_name, total, process_fn, delay=3, batch_pause=10):
    """Generic phase runner with resume + rate limiting."""
    data = load(file_name) or []
    start = len(data)
    if start >= total:
        print(f"\n  {phase_name}: Loaded {len(data)} cached results", flush=True)
        return data

    print(f"\n  {phase_name}: {start}/{total} done, {total - start} remaining...", flush=True)
    for idx in range(start, total):
        result = process_fn(idx)
        data.append(result)

        if (idx + 1) % 5 == 0:
            print(f"    {idx+1:>3}/{total}", flush=True)

        time.sleep(delay)
        if (idx + 1) % batch_pause == 0:
            save(data, file_name)
            print(f"    [saved {idx+1}/{total}]", flush=True)
            time.sleep(5)

    save(data, file_name)
    print(f"    Saved {phase_name} ({len(data)} items)", flush=True)
    return data


def main():
    print("=" * 60, flush=True)
    print("  EVALUATION HARNESS", flush=True)
    print("=" * 60, flush=True)

    import random
    with open(os.path.join(DATA_DIR, 'golden_eval_set.json')) as f:
        all_golden = json.load(f)

    # Stratified sample: pick SAMPLE_SIZE examples with intent diversity
    random.seed(42)
    golden = random.sample(all_golden, min(SAMPLE_SIZE, len(all_golden)))
    N = len(golden)
    print(f"  Sampled {N} from {len(all_golden)} golden examples", flush=True)

    # ── PHASE 1: Classify ──
    def classify(idx):
        tweet = golden[idx]
        resp = call(messages=[{"role": "user", "content":
            f"You are an intent classifier for Spotify support.\n"
            f"Classify into ONE intent:\n{INTENT_DESCRIPTIONS}\n"
            f"User: \"{tweet['text']}\"\n"
            f"Respond with ONLY the label. /no_think"}],
            temperature=0, max_tokens=30)
        if not resp:
            return "other"
        raw = strip_think(resp.choices[0].message.content).lower()
        for i in INTENTS:
            if i in raw:
                return i
        return "other"

    phase1 = run_phase("Phase 1 (classify)", "eval_phase1.json", N, classify, delay=4, batch_pause=15)

    acc = sum(1 for i in range(N) if phase1[i] == golden[i]['gold_intent'])
    print(f"    Intent accuracy: {acc}/{N} ({acc/N:.1%})", flush=True)

    # ── PHASE 2: Retrieve (local) ──
    phase2 = load("eval_phase2.json")
    if phase2 and len(phase2) == N:
        print(f"\n  Phase 2 (retrieve): Loaded {N} cached", flush=True)
    else:
        print(f"\n  Phase 2 (retrieve): Running locally...", flush=True)
        retriever = KBRetriever(load_existing=True)

        import chromadb
        from sentence_transformers import SentenceTransformer
        cc = chromadb.PersistentClient(path=os.path.join(KB_DIR, 'chroma_db'))
        hist = cc.get_collection("spotify_historical")
        emb = SentenceTransformer("all-MiniLM-L6-v2")

        with open(os.path.join(KB_DIR, 'diagnostics.json')) as f:
            diag_data = json.load(f)

        phase2 = []
        for idx in range(N):
            intent = phase1[idx]
            text = golden[idx]['text']

            kb = retriever.retrieve(text, intent=intent, top_k=2)

            qe = emb.encode([text]).tolist()
            try:
                hr = hist.query(query_embeddings=qe, n_results=3, where={"intent": intent})
            except Exception:
                hr = hist.query(query_embeddings=qe, n_results=3)

            hist_results = []
            if hr['documents'] and hr['documents'][0]:
                for i, doc in enumerate(hr['documents'][0]):
                    d = hr['distances'][0][i] if hr['distances'] else 0
                    hist_results.append({'score': round(1 - d, 4), 'text': doc})

            diag = diag_data.get('per_intent', {}).get(intent, diag_data.get('standard_flow', []))

            phase2.append({
                'kb': kb, 'hist': hist_results, 'diag': diag,
                'kb_topic': kb[0]['topic'] if kb else 'none',
                'kb_score': kb[0]['score'] if kb else 0,
            })

        save(phase2, "eval_phase2.json")
        print(f"    Saved Phase 2 (0 API calls)", flush=True)

    # ── PHASE 3: Generate responses ──
    def respond(idx):
        intent = phase1[idx]
        r = phase2[idx]
        text = golden[idx]['text']

        kb_ctx = "".join(f"\n--- {x['topic']} ---\n{x['text']}\n" for x in r['kb'][:2])
        hist_ctx = "".join(f"\n--- Historical ---\n{x['text']}\n" for x in r['hist'][:3])
        diag_txt = "\n".join(f"  - {q}" for q in r['diag'])

        resp = call(messages=[
            {"role": "system", "content":
                f"You are SpotBot, a Spotify support bot. Be concise (4-5 sentences). "
                f"Give numbered steps. Warm casual tone. Never make up info. "
                f"For account issues, direct to support.spotify.com/contact-spotify-support/ "
                f"Learn from historical conversations. /no_think\n"
                f"Intent: {intent}\nKB:{kb_ctx}\nHistorical:{hist_ctx}\nDiagnostics:\n{diag_txt}"},
            {"role": "user", "content": text}],
            temperature=0.3, max_tokens=300)
        if not resp:
            return "Sorry, I'm having trouble right now."
        return strip_think(resp.choices[0].message.content)

    phase3 = run_phase("Phase 3 (respond)", "eval_phase3.json", N, respond, delay=4, batch_pause=15)

    # ── PHASE 4: Judge responses ──
    def judge(idx):
        text = golden[idx]['text']
        bot = phase3[idx]
        kb_text = phase2[idx]['kb'][0]['text'][:400] if phase2[idx]['kb'] else ""

        resp = call(messages=[{"role": "user", "content":
            f"Score this support bot response 1-5 on 4 dimensions.\n"
            f"CUSTOMER: \"{text}\"\n"
            f"BOT: \"{bot}\"\n"
            f"KB: \"{kb_text}\"\n\n"
            f"factual: accuracy per KB (5=perfect,1=wrong)\n"
            f"actionability: clear steps? (5=specific,1=none)\n"
            f"tone: SpotifyCares style? (5=warm,1=robotic)\n"
            f"diagnostic: right approach? (5=perfect,1=wrong)\n\n"
            f"Reply ONLY JSON: {{\"factual\":X,\"actionability\":X,\"tone\":X,\"diagnostic\":X}} /no_think"}],
            temperature=0, max_tokens=50)

        default = {"factual": 3, "actionability": 3, "tone": 3, "diagnostic": 3}
        if not resp:
            return default
        raw = strip_think(resp.choices[0].message.content)
        try:
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if m:
                parsed = json.loads(m.group())
                return {k: int(parsed.get(k, 3)) for k in default}
        except Exception:
            pass
        return default

    phase4 = run_phase("Phase 4 (judge)", "eval_phase4.json", N, judge, delay=4, batch_pause=15)

    # ── COMPILE ──
    print(f"\n  Compiling results...", flush=True)
    from collections import defaultdict

    results = []
    ci, ce = 0, 0
    tj = {"factual": 0, "actionability": 0, "tone": 0, "diagnostic": 0}

    for idx in range(N):
        g = golden[idx]
        ic = phase1[idx] == g['gold_intent']
        if ic: ci += 1

        ge = gold_escalate(g['gold_intent'], g['text'])
        pe = any(k in phase3[idx].lower() for k in ['support.spotify.com/contact', 'specialist', 'escalat'])
        ec = ge == pe
        if ec: ce += 1

        for d in tj:
            tj[d] += phase4[idx].get(d, 3)

        results.append({
            'text': g['text'], 'gold_intent': g['gold_intent'],
            'pred_intent': phase1[idx], 'intent_correct': ic,
            'gold_escalate': ge, 'pred_escalate': pe, 'escalation_correct': ec,
            'bot_response': phase3[idx],
            'kb_topic': phase2[idx]['kb_topic'], 'kb_score': phase2[idx]['kb_score'],
            'judge_scores': phase4[idx], 'difficulty': g.get('difficulty', ''),
        })

    avg_j = {d: tj[d] / N for d in tj}

    pi = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    for r in results:
        if r['gold_intent'] == r['pred_intent']:
            pi[r['gold_intent']]['tp'] += 1
        else:
            pi[r['pred_intent']]['fp'] += 1
            pi[r['gold_intent']]['fn'] += 1

    pf1 = {}
    for intent in sorted(pi):
        tp, fp, fn = pi[intent]['tp'], pi[intent]['fp'], pi[intent]['fn']
        p = tp / (tp + fp) if tp + fp else 0
        r = tp / (tp + fn) if tp + fn else 0
        pf1[intent] = {'p': round(p, 3), 'r': round(r, 3),
                        'f1': round(2*p*r/(p+r) if p+r else 0, 3), 'n': tp + fn}

    da = {}
    for diff in ['easy', 'medium', 'hard']:
        dr = [r for r in results if r['difficulty'] == diff]
        if dr:
            da[diff] = {'n': len(dr),
                        'intent_acc': round(sum(r['intent_correct'] for r in dr) / len(dr), 3),
                        'avg_judge': round(sum(sum(r['judge_scores'].values())/4 for r in dr) / len(dr), 2)}

    summary = {
        'intent_accuracy': round(ci / N, 4),
        'escalation_accuracy': round(ce / N, 4),
        'avg_judge': {k: round(v, 2) for k, v in avg_j.items()},
        'overall_judge': round(sum(avg_j.values()) / 4, 2),
        'per_intent_f1': pf1, 'by_difficulty': da,
    }

    save(results, 'eval_results.json')
    save(summary, 'eval_summary.json')

    # Print
    print(f"\n{'=' * 60}", flush=True)
    print(f"  RESULTS", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"  Intent:     {ci}/{N} ({ci/N:.1%})", flush=True)
    print(f"  Escalation: {ce}/{N} ({ce/N:.1%})", flush=True)
    print(f"\n  Judge (1-5):", flush=True)
    for d, s in avg_j.items():
        print(f"    {d:<20} {s:.2f}", flush=True)
    print(f"    {'OVERALL':<20} {sum(avg_j.values())/4:.2f}", flush=True)
    print(f"\n  By difficulty:", flush=True)
    for d, s in da.items():
        print(f"    {d:<10} intent:{s['intent_acc']:.1%} judge:{s['avg_judge']} (n={s['n']})", flush=True)
    print(f"\n  Per-intent F1:", flush=True)
    for i, s in sorted(pf1.items(), key=lambda x: x[1]['f1'], reverse=True):
        print(f"    {i:<25} F1={s['f1']:.3f} P={s['p']:.3f} R={s['r']:.3f} (n={s['n']})", flush=True)

    fails = [r for r in results if not r['intent_correct']]
    print(f"\n  Failures ({len(fails)}):", flush=True)
    for f in fails[:15]:
        print(f"    gold:{f['gold_intent']:<22} pred:{f['pred_intent']:<22} \"{f['text'][:50]}\"", flush=True)

    print(f"\n  Saved → eval_results.json + eval_summary.json", flush=True)


if __name__ == '__main__':
    main()
