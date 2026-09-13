"""
Script 3: Build the knowledge base

Combines:
  1. Scraped Spotify help center articles
  2. Resolution patterns extracted from SpotifyCares tweets in the CSV
  3. Diagnostic question flows from tweets

Outputs:
  - knowledge_base/kb.json           (full structured KB)
  - knowledge_base/diagnostics.json  (per-intent diagnostic flows)

"""

import csv
import json
import os
import re
from collections import Counter, defaultdict

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
KB_DIR = os.path.join(BASE_DIR, 'knowledge_base')
CSV_PATH = os.path.join(BASE_DIR, '..', 'twcs.csv')


# ── Part A: Scraped help articles (from Spotify support site) ──

SCRAPED_ARTICLES = {
    "subscription_billing": [
        {
            "topic": "Cancel Premium",
            "url": "https://support.spotify.com/us/article/cancel-premium/",
            "steps": [
                "Go to your account page at spotify.com/account",
                "Click 'Manage your plan'",
                "Select 'Cancel subscription'",
                "Your Premium continues until the next billing date, then converts to free"
            ],
            "notes": [
                "Canceling during a free trial switches you to free immediately",
                "Playlists and saved music stay after downgrading",
                "Family/Duo members: leaving the plan doesn't cancel it — contact the plan manager",
                "If no cancel option appears, your sub is through a partner (check Payment section for their contact)"
            ]
        },
        {
            "topic": "Change Payment Details",
            "url": "https://support.spotify.com/us/article/change-payment-details/",
            "steps": [
                "Go to spotify.com/account",
                "Navigate to 'Payment methods'",
                "Select 'Add' for new or 'Change' for existing methods",
                "Confirm with 'Add/Change payment method'"
            ],
            "notes": [
                "A small temporary authorization charge may appear — it disappears shortly",
                "To remove a saved card: Payment methods > Saved payment methods > Remove"
            ]
        },
        {
            "topic": "Failed Payment",
            "url": "https://support.spotify.com/us/article/failed-payment/",
            "steps": [
                "Retry payment in a private/incognito browser window",
                "Try a different payment method",
                "Wait a few hours — may be a temporary connection issue",
                "Contact your bank or payment provider directly"
            ],
            "requirements": [
                "Payment method must have sufficient funds",
                "Must be registered in the same country as your Spotify account",
                "Must not be expired or canceled",
                "Must be enabled for foreign, secure online, and recurring purchases"
            ],
            "notes": [
                "You won't immediately lose Premium if a payment fails — Spotify retries over the next few days"
            ]
        },
        {
            "topic": "Premium Plans",
            "url": "https://spotify.com/us/premium/",
            "plans": {
                "Individual": "$12.99/month — 1 account, ad-free, lossless, offline, 15 audiobook hrs",
                "Student": "$6.99/month — 1 account, all Individual features + Hulu",
                "Duo": "$18.99/month — 2 accounts, same address required",
                "Family": "$21.99/month — up to 6 accounts, parental controls, same address required"
            }
        }
    ],
    "device_compatibility": [
        {
            "topic": "Spotify Connect",
            "url": "https://support.spotify.com/us/article/spotify-connect/",
            "steps": [
                "Connect all devices to the same WiFi network",
                "Open Spotify and start playing",
                "Tap the connect/device icon at the bottom",
                "Select your target device"
            ],
            "troubleshooting": [
                "Device not showing? Toggle 'Local device visibility' in Settings > Apps and devices",
                "iPhone users: grant local network access in iOS Settings > Spotify > Local Network",
                "Restart the Spotify app and target device",
                "Make sure all software is up to date",
                "Try resetting WiFi or switching networks",
                "Android TV: enable 'display over other apps' in TV Settings > Apps > Special app access",
                "If paused for 10+ minutes, you may need to reconnect"
            ]
        },
        {
            "topic": "Bluetooth",
            "url": "https://support.spotify.com/us/article/bluetooth/",
            "steps": [
                "Close Spotify first",
                "Turn on Bluetooth on both devices",
                "Pair them in Bluetooth settings",
                "Open Spotify and play"
            ],
            "troubleshooting": [
                "Keep devices within 1 meter / 3 feet of each other",
                "Disconnect other Bluetooth devices (some only allow one connection)",
                "Check that Bluetooth allows media sharing in device settings",
                "Make sure both devices have enough battery",
                "Re-pair: Forget/Unpair the device > Bluetooth off then on > reconnect"
            ]
        },
        {
            "topic": "Spotify on PlayStation",
            "url": "https://support.spotify.com/us/article/spotify-on-playstation/",
            "supported": "PS5 and PS4",
            "login_methods": [
                "Enter Spotify email and password directly",
                "Use Spotify Connect from your phone/tablet",
                "Select LOG IN WITH PIN > visit spotify.com/pair on another device"
            ],
            "controls": {
                "PS5": "Press PS button > select Music from control center",
                "PS4": "Press and hold PS button > select Spotify"
            },
            "disconnect": [
                "Web: spotify.com/account > Manage apps > Remove Access for Sony",
                "Console: Settings > Users and Accounts > Link Services > Spotify > Unlink"
            ],
            "tip": "Disable in-game music for the best experience while gaming"
        },
        {
            "topic": "Spotify on Speakers",
            "url": "https://support.spotify.com/us/article/spotify-on-speakers/",
            "connection_methods": [
                "Spotify Connect", "Voice Assistants", "Bluetooth",
                "AUX or USB cable", "Google Chromecast Audio", "Apple AirPlay"
            ],
            "troubleshooting": [
                "Update app and device software",
                "Restart Spotify app",
                "Close unused apps",
                "Restart speaker and other devices",
                "Restart WiFi router",
                "Try a different WiFi network",
                "For cables: try a different cable"
            ]
        }
    ],
    "playback_issue": [
        {
            "topic": "Playback Troubleshooting",
            "steps": [
                "Force close Spotify and reopen it",
                "Log out > restart device > log back in",
                "Check internet connection (switch between WiFi and mobile data)",
                "Check if Spotify is down (status page or downdetector)"
            ],
            "device_specific": {
                "iPhone": "Restart by holding Sleep/Wake + Volume Down for 10 seconds",
                "Android": "Force stop: Settings > Apps > Spotify > Force Stop",
                "Desktop": "Close app completely (check system tray) and relaunch"
            },
            "reinstall": [
                "Uninstall Spotify completely",
                "Restart your device",
                "Reinstall from App Store / Play Store / spotify.com",
                "Log back in (offline downloads will need re-downloading)"
            ],
            "common_causes": [
                "Another app taking audio focus (Snapchat, phone calls, navigation)",
                "Outdated app version",
                "Low device storage",
                "Corrupted cache"
            ]
        }
    ],
    "account_access": [
        {
            "topic": "Protect Your Account",
            "url": "https://support.spotify.com/us/article/protect-your-account/",
            "password_tips": [
                "Use long passwords with letters, numbers, and special characters",
                "Use a different password for each service",
                "Change your password regularly",
                "Never share your password"
            ],
            "sign_out_everywhere": [
                "Go to spotify.com/account",
                "Under Security and privacy, select 'Sign out everywhere'",
                "Confirm the action",
                "Note: this doesn't cover speakers/consoles/TVs — remove those via Manage apps"
            ],
            "if_hacked": [
                "Reset your password immediately at spotify.com/password-reset",
                "Sign out everywhere from your account page",
                "Remove unrecognized third-party apps",
                "Check your email address hasn't been changed",
                "Contact Spotify support if you can't access your account at all"
            ]
        }
    ],
    "shuffle_queue": [
        {
            "topic": "Shuffle",
            "url": "https://support.spotify.com/us/article/shuffle/",
            "how_to": "Tap the shuffle icon to toggle. Off = plays in order. On = plays randomly.",
            "styles_premium": {
                "Fewer Repeats": "Tracks what you've played recently, picks from multiple random sequences. Reduces repetition.",
                "Standard": "Pure random — every track has equal chance. May repeat sooner."
            },
            "smart_shuffle": {
                "what": "Mixes in recommended songs matching the playlist vibe (marked with Enhance badge)",
                "free_mobile": "Always active",
                "premium": "Toggle via shuffle icon in playlists",
                "disable": "Profile > Settings and privacy > Playback > toggle off 'Include Smart Shuffle in play modes'"
            }
        },
        {
            "topic": "Play Queue",
            "url": "https://support.spotify.com/us/article/play-queue/",
            "mobile": {
                "open": "Tap Now Playing bar > tap queue icon",
                "add": "Tap ··· next to a track > 'Add to Queue'",
                "reorder": "Tap and hold drag icon > drag to position",
                "remove": "Swipe left on the track",
                "clear": "Tap queue icon > 'Clear'"
            },
            "desktop": {
                "open": "Click queue icon near playback controls",
                "add": "Right-click track > 'Add to queue'",
                "reorder": "Click and drag to reposition",
                "remove": "Right-click > remove",
                "clear": "'Clear queue' > confirm 'Yes'"
            }
        }
    ],
    "audio_quality": [
        {
            "topic": "Audio Quality Settings",
            "url": "https://support.spotify.com/us/article/audio-quality/",
            "quality_tiers": {
                "Free": {
                    "Web": "AAC 128kbit/s (fixed)",
                    "App": "Auto | Low ~24kbit/s | Normal ~96kbit/s | High ~160kbit/s"
                },
                "Premium": {
                    "Web": "AAC 256kbit/s (fixed)",
                    "App": "Auto | Low ~24kbit/s | Normal ~96kbit/s | High ~160kbit/s | Very High ~320kbit/s | Lossless 24-bit/44.1kHz FLAC"
                }
            },
            "how_to_change": {
                "Mobile/Tablet": "Profile pic > Settings and privacy > Media Quality (separate WiFi vs cellular settings)",
                "Desktop": "Profile pic > Settings > Audio Quality",
                "Web Player": "Cannot change — download the app for quality options"
            },
            "tips": [
                "Auto-adjust is on by default to prevent stuttering — turn it off for manual control",
                "Podcasts stream at ~96kbit/s (128 on web). Low quality drops them to ~24kbit/s",
                "Lossless requires Premium and a supported device"
            ]
        }
    ],
    "ads_complaints": [
        {
            "topic": "Ads on Spotify Free",
            "how_ads_work": "Spotify Free includes audio and display ads between songs to support free access.",
            "reduce_ads": [
                "Upgrade to Premium ($12.99/month Individual, $6.99/month Student) for ad-free listening",
                "Watch a video ad when offered for 30 minutes of ad-free listening (mobile only)"
            ],
            "common_issues": {
                "30min_not_working": "If the app crashes during the 30-minute window, the timer resets. Watch the full video ad and keep the app open.",
                "too_many_ads": "Ad frequency is set by Spotify and can't be customized on the free tier.",
                "inappropriate_ad": "Report inappropriate ads through the app or contact support."
            },
            "note": "There is no setting to reduce ad frequency or customize ads on the free tier."
        }
    ],
    "content_availability": [
        {
            "topic": "Why Content Is Missing",
            "reasons": [
                "Licensing agreements vary by country/region",
                "Artists or labels may remove their own content",
                "Content can be temporarily unavailable during licensing negotiations",
                "Some tracks are region-locked"
            ],
            "what_to_do": [
                "Check for alternate versions or compilations",
                "Follow the artist to get notified of new releases",
                "Greyed out tracks = exists but not available in your region",
                "Request content via the Spotify Community forums"
            ]
        }
    ],
    "download_offline": [
        {
            "topic": "Downloads Unexpectedly Removed",
            "causes": [
                "Using more offline devices than allowed",
                "Not going online at least once every 30 days",
                "App update or reinstall cleared cached downloads",
                "Low storage space on device"
            ],
            "fixes": [
                "Check and manage offline devices at spotify.com/account",
                "Go online at least once every 30 days",
                "Free up storage space, then re-download",
                "If persistent: reinstall the app and re-download"
            ],
            "limits": [
                "Up to 10,000 tracks per device",
                "Up to 5 offline devices",
                "Must connect to internet every 30 days"
            ],
            "how_to_download": [
                "Open a playlist, album, or podcast",
                "Tap the download toggle/arrow icon",
                "Green arrow = downloaded successfully",
                "Offline mode: Settings > Playback > Offline mode"
            ]
        }
    ],
    "country_region": [
        {
            "topic": "Spotify Availability",
            "info": "Spotify launches in new countries regularly but is not yet available worldwide.",
            "what_to_do": [
                "Sign up at spotify.com to be notified when available in your country",
                "Check spotify.com for the list of supported countries"
            ],
            "traveling": "You can use Spotify abroad for up to 14 days. After that, update your country setting in account settings."
        }
    ],
    "feature_request": [
        {
            "topic": "How to Request Features",
            "steps": [
                "Visit the Spotify Community Ideas board at community.spotify.com",
                "Search if your idea already exists",
                "If it exists, vote for it",
                "If not, create a new idea post",
                "Popular ideas are reviewed by the Spotify team"
            ],
            "bot_response_template": "Thanks for the suggestion! You can vote for this on the Spotify Community Ideas board — the more votes, the more likely it'll be considered."
        }
    ]
}


def extract_tweet_patterns(csv_path):
    """Extract resolution patterns and diagnostic questions from SpotifyCares tweets."""

    resolution_examples = defaultdict(list)
    diagnostic_questions = defaultdict(list)

    resolution_markers = [
        'try ', 'go to ', 'tap ', 'click ', 'select ', 'open ',
        'log out', 'restart', 'reinstall', 'uninstall', 'update ',
        'clear cache', 'settings >', 'enable', 'disable', 'toggle',
        "here's how", 'you can ', "you'll need", 'make sure',
    ]
    diagnostic_markers = [
        'what device', 'which device', 'operating system',
        'spotify version', 'what version', 'what country',
        'what plan', 'premium or free', 'wifi or', '3g/4g',
        'does this happen', 'does restarting', 'how long',
        'when did this start', 'can you try', 'does it happen',
        'can you confirm', 'can you let us know',
    ]

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['author_id'].strip() != 'SpotifyCares':
                continue
            text = row['text'].strip()
            text_lower = text.lower()

            for marker in resolution_markers:
                if marker in text_lower:
                    if len(resolution_examples[marker]) < 3:
                        resolution_examples[marker].append(text[:200])
                    break

            for marker in diagnostic_markers:
                if marker in text_lower:
                    if len(diagnostic_questions[marker]) < 3:
                        diagnostic_questions[marker].append(text[:200])
                    break

    return resolution_examples, diagnostic_questions


def extract_url_intent_mapping(csv_path):
    """Map which URLs SpotifyCares shares for each intent."""
    tweets = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tweets[row['tweet_id'].strip()] = row

    intent_patterns = {
        'playback_issue': ['not playing', "won't play", 'stops playing', 'crashes', 'freezes', 'not working'],
        'account_access': ["can't log in", 'password', 'hacked', "can't access", 'locked out'],
        'subscription_billing': ['charged', 'billing', 'payment', 'refund', 'cancel', 'premium', 'subscription'],
        'content_availability': ['not available', 'missing', 'removed', "can't find", 'greyed out'],
        'download_offline': ['download', 'offline', 'downloads disappeared'],
        'shuffle_queue': ['shuffle', 'queue', 'repeat', 'play in order'],
        'device_compatibility': ['bluetooth', 'chromecast', 'alexa', 'car', 'roku', 'ps4', 'xbox', 'connect'],
        'ads_complaints': ['ads', 'ad', 'advertisement', 'too many ads'],
        'audio_quality': ['sound quality', 'audio quality', 'bitrate', 'distorted', 'lossless'],
        'playlist_library': ['playlist', 'library', 'discover weekly', 'daily mix'],
    }

    intent_urls = defaultdict(Counter)

    for tid, t in tweets.items():
        if t['inbound'].strip() != 'True':
            continue
        text_lower = t['text'].strip().lower()

        matched_intent = None
        for intent, keywords in intent_patterns.items():
            if any(kw in text_lower for kw in keywords):
                matched_intent = intent
                break
        if not matched_intent:
            continue

        resp_ids = [r.strip() for r in t['response_tweet_id'].strip().split(',') if r.strip()]
        for rid in resp_ids:
            if rid in tweets and tweets[rid]['author_id'].strip() == 'SpotifyCares':
                urls = re.findall(r'https?://t\.co/\S+', tweets[rid]['text'])
                for u in urls:
                    clean = re.sub(r'[.,;:!?\)\]]+$', '', u)
                    intent_urls[matched_intent][clean] += 1

    return intent_urls


def main():
    os.makedirs(KB_DIR, exist_ok=True)

    print("Building knowledge base...\n")

    # Part A: Scraped articles already defined above
    print("Part A: Scraped help articles")
    for intent, articles in SCRAPED_ARTICLES.items():
        topic_count = len(articles) if isinstance(articles, list) else 1
        print(f"  {intent:<25} {topic_count} article(s)")

    # Part B: Extract tweet patterns
    print("\nPart B: Extracting resolution patterns from tweets...")
    resolutions, diagnostics = extract_tweet_patterns(CSV_PATH)
    print(f"  Found {len(resolutions)} resolution patterns")
    print(f"  Found {len(diagnostics)} diagnostic question patterns")

    # Part C: URL-to-intent mapping
    print("\nPart C: Mapping URLs to intents...")
    intent_urls = extract_url_intent_mapping(CSV_PATH)
    for intent, urls in sorted(intent_urls.items()):
        top3 = urls.most_common(3)
        print(f"  {intent:<25} {len(urls)} unique URLs, top: {top3[0][0] if top3 else 'none'}")

    # Build final KB
    kb = {
        "version": "1.0",
        "brand": "SpotifyCares",
        "intents": {}
    }

    for intent, articles in SCRAPED_ARTICLES.items():
        kb["intents"][intent] = {
            "articles": articles,
            "top_urls": dict(intent_urls.get(intent, Counter()).most_common(10)),
        }

    kb_path = os.path.join(KB_DIR, 'kb.json')
    with open(kb_path, 'w', encoding='utf-8') as f:
        json.dump(kb, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved knowledge base   -> {kb_path}")

    # Build diagnostics file
    diag_flows = {
        "standard_flow": [
            "What device are you using? (iPhone, Android, Desktop, PS4, etc.)",
            "What operating system and version?",
            "What Spotify version? (found in Settings > About)",
            "Are you on WiFi or mobile data?",
            "Are you Free or Premium?",
            "When did this issue start?",
            "Does restarting the app help?"
        ],
        "per_intent": {
            "playback_issue": [
                "What device and OS are you using?",
                "What Spotify version?",
                "Does this happen on WiFi, mobile data, or both?",
                "Does restarting the device help?",
                "Does it happen with all songs or specific ones?"
            ],
            "device_compatibility": [
                "What device are you trying to connect to?",
                "What phone/computer are you controlling from?",
                "Are both devices on the same WiFi network?",
                "What Spotify version are you using?",
                "Have you tried unpairing and re-pairing?"
            ],
            "account_access": [
                "What error message do you see when trying to log in?",
                "Are you using email, Facebook, Apple, or Google to log in?",
                "Have you tried resetting your password?",
                "Do you have access to the email on your account?"
            ],
            "subscription_billing": [
                "What plan are you on? (Individual, Student, Duo, Family)",
                "What payment method are you using?",
                "What country is your account registered in?",
                "Can you check your payment history at spotify.com/account?"
            ],
            "download_offline": [
                "What device are you downloading on?",
                "How many offline devices do you have?",
                "When did the downloads disappear?",
                "Do you have enough storage space on your device?"
            ],
            "shuffle_queue": [
                "What device are you using?",
                "Are you Free or Premium?",
                "Are you trying to play a playlist, album, or liked songs?",
                "What happens when you tap the shuffle icon?"
            ],
            "audio_quality": [
                "What quality setting do you have selected?",
                "What device and connection type (WiFi/mobile)?",
                "Does this happen with all songs or specific ones?",
                "Are you Free or Premium?"
            ],
            "ads_complaints": [
                "Are you on the Free or Premium plan?",
                "What device are you using?",
                "Did you watch a video ad for 30-min ad-free? Did the app crash during that time?"
            ],
            "playlist_library": [
                "What device and Spotify version are you using?",
                "Which playlist is affected?",
                "When did you notice the issue?",
                "Have you tried logging out and back in?"
            ],
            "content_availability": [
                "What song/album/artist are you looking for?",
                "What country is your account set to?",
                "Is the content greyed out or completely missing?"
            ],
            "country_region": [
                "What country are you in?",
                "Have you checked if Spotify is available in your country at spotify.com?"
            ]
        },
        "resolution_examples_from_tweets": {k: v for k, v in resolutions.items()},
        "diagnostic_examples_from_tweets": {k: v for k, v in diagnostics.items()},
    }

    diag_path = os.path.join(KB_DIR, 'diagnostics.json')
    with open(diag_path, 'w', encoding='utf-8') as f:
        json.dump(diag_flows, f, indent=2, ensure_ascii=False)
    print(f"  Saved diagnostics      -> {diag_path}")

    # Print coverage summary
    print(f"\n--- Knowledge Base Coverage ---\n")
    print(f"  {'Intent':<25} {'Articles':>8} {'URLs':>8} {'Diagnostics':>12}")
    print(f"  {'-'*55}")
    all_intents = set(list(SCRAPED_ARTICLES.keys()) + list(intent_urls.keys()))
    for intent in sorted(all_intents):
        a = len(SCRAPED_ARTICLES.get(intent, []))
        u = len(intent_urls.get(intent, {}))
        d = 'YES' if intent in diag_flows['per_intent'] else 'NO'
        print(f"  {intent:<25} {a:>8} {u:>8} {d:>12}")


if __name__ == '__main__':
    main()
