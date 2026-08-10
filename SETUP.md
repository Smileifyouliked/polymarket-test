# Setup guide

Written for someone who has not done this before. Every command is meant to be
copy-pasted. After each one there's what success looks like, so you can tell
whether to carry on or stop.

**Read this first:** the model has never seen a real CS2 match. Everything
proven so far is on a simulated league. Part 3 is where that changes, and
Part 4 is where you find out whether it was worth it.

---

## Part 0 — What you're setting up

| Piece | Status | What it does |
|---|---|---|
| Forecaster | built, tested | Rates teams per map, simulates the veto, outputs a calibrated probability |
| Bankroll tracker | built, tested | Records positions, tracks capital, P&L, win rate, exposure |
| Heartbeat + dashboard | built, tested | Tells you it's alive and whether you're up or down |
| Polymarket bot | built, tested | Discovers markets, prices them, risk-checks, trades, settles |
| Risk limits + kill switch | built, tested | Edge/exposure/loss caps; `cli stop` halts it instantly |
| Real data ingest | **unverified** | Written but never run against the live API |
| Polymarket calls | **unverified** | Written against the real client, but never sent |
| Upcoming fixtures | **not built** | Can't yet tell you what's playing tomorrow |

It is a **pre-match** forecaster. It does not watch rounds live. It runs in
DRY RUN unless you pass `--live`, and the model has never been validated on a
real CS2 match — Part 4 is where you find out whether it is worth anything.

---

## Part 1 — The server

### 1.1 Connect

```bash
ssh -i /path/to/your-key.pem ubuntu@YOUR_SERVER_IP
```

Username depends on the AMI: Ubuntu → `ubuntu`, Amazon Linux → `ec2-user`,
Debian → `admin`.

| Problem | Fix |
|---|---|
| `Permission denied (publickey)` | Wrong username — try the others |
| `UNPROTECTED PRIVATE KEY FILE` | `chmod 400 /path/to/your-key.pem` |
| Hangs, then times out | Security group blocks SSH. AWS console → EC2 → your instance → Security → Security groups → Edit inbound rules → Add rule → Type `SSH`, Source `My IP` |

✅ Prompt changes to something like `ubuntu@ip-172-31-0-1:~$`

### 1.2 Install prerequisites

```bash
# Ubuntu / Debian
sudo apt update && sudo apt install -y python3 python3-venv python3-pip git tmux

# Amazon Linux
sudo dnf install -y python3 python3-pip git tmux
```

```bash
python3 --version
```

✅ `Python 3.9` or higher.

### 1.3 Get the code

```bash
git clone https://github.com/Smileifyouliked/polymarket-test.git
cd polymarket-test
git checkout claude/polymarket-github-repos-xomddj
ls
```

✅ You see `cs2model`, `tests`, `SETUP.md`, `README_CS2.md`.

### 1.4 Install into a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-cs2.txt
```

✅ Prompt now starts with `(venv)`, and pip ends with `Successfully installed`.

> **The single most common mistake:** every time you reconnect to the server
> you must re-run these two lines before anything else works:
> ```bash
> cd polymarket-test && source venv/bin/activate
> ```
> If a command fails with `ModuleNotFoundError: No module named 'numpy'`, this
> is why.

**If `pip install` gets Killed** (t2.micro / t3.micro have only 1 GB RAM), add
swap once and retry:

```bash
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
```

---

## Part 2 — Prove it works (no API key needed)

```bash
python3 -m pytest tests/ -q
```

✅ `79 passed`

```bash
python3 -m cs2model.cli demo
```

✅ Takes ~30s and prints a baseline table, a coverage curve and a reliability
table. **If you get here, your server is set up correctly.**

Everything above ran on simulated data. Now for the real thing.

---

## Part 3 — Real data

### 3.1 The API key

Go to **https://liquipedia.net/api-terms-of-use** and follow their signup.
It's free.

```bash
echo 'export LIQUIPEDIA_API_KEY=paste_your_key_here' >> ~/.bashrc
echo 'export LIQUIPEDIA_USER_AGENT="cs2model/0.1 (your@email.com)"' >> ~/.bashrc
source ~/.bashrc
echo $LIQUIPEDIA_API_KEY
```

✅ Your key prints back.

The User-Agent is not optional politeness — Liquipedia asks for real contact
info and blocks clients that don't identify themselves.

**Never** paste the key into a file in this repo, a commit, or a chat message.
`~/.bashrc` is outside the repo, which is exactly why it goes there.

### 3.2 The step that everything depends on

```bash
python3 -m cs2model.cli explore --limit 3 --save data/raw_finished.json
```

This dumps the API's raw response. I wrote the parser from documentation, not
from seeing real data, so **the field names are educated guesses.** This tells
us whether they're right.

✅ Prints `API base URL that answered: ...`, then a list of field names, then
one full record.

| Error | Meaning |
|---|---|
| `LIQUIPEDIA_API_KEY is not set` | Step 3.1 didn't take — reopen the terminal |
| `HTTP 401` | Key is wrong or not yet active |
| `HTTP 429` | Rate limited — wait a few minutes. Don't lower `SLEEP_SECONDS` |
| `Could not find a working base URL` | Paste me the whole message |

### 3.3 Also grab an upcoming match

```bash
python3 -m cs2model.cli explore --limit 3 --conditions "[[winner::]]" \
    --save data/raw_upcoming.json
```

This asks for matches with no winner yet — i.e. fixtures that haven't been
played. It's what the not-yet-built "what's on tomorrow" feature needs.

**Send me both files.** They contain match data only, no credentials.

### 3.4 Ingest

Only once 3.2 looks right:

```bash
python3 -m cs2model.cli ingest --out data/matches.json --pages 40
```

Takes a few minutes — the client deliberately sleeps 2.2s between pages to
stay inside the rate limit.

✅ `ingested N usable matches, skipped M unusable records`

If N is 0, or M dwarfs N, the field mapping is wrong. Send me the output.

---

## Part 4 — Is the model actually any good?

```bash
python3 -m cs2model.cli evaluate --data data/matches.json
```

**This is a real decision point, not a formality.** Read the baseline block:

```
always-pick-favourite  acc= 0.679
glicko-only            acc= 0.679  logloss= 0.594
veto-structural        acc= 0.690  logloss= 0.583
FULL MODEL             acc= 0.688  logloss= 0.586
```

- **FULL MODEL beats the others** → good, carry on.
- **It doesn't** → the model isn't earning its keep on real data. Better to
  learn that here than after forty bets. Tell me and we'll dig in.

Then read the coverage curve, which is the answer to "how accurate is it?":

```
  confidence   accuracy   coverage
        0.60      0.734     73.3%
        0.70      0.800     40.8%
```

At confidence 0.70 it's right 80% of the time but only calls 4 matches in 10.
**Pick the row you're happy with — that threshold is your operating setting.**

Expect worse numbers than the demo. The demo runs on a simulated league I
tuned; real Counter-Strike is messier.

---

## Part 5 — Predict a match

```bash
python3 -m cs2model.cli predict --data data/matches.json \
    --team-a "Vitality" --team-b "Spirit" --bo 3 --lan --threshold 0.70
```

Use the threshold you chose in Part 4.

You get per-map probabilities, how often each map survives the veto, and either
a verdict or `NO CALL`. **`NO CALL` is the model working, not failing** — it's
declining a coinflip, which is exactly what buys the accuracy.

Team names must match the data. If it says `unknown team`, it prints examples
of names it does know.

---

## Part 6 — Paper trade first

Use real odds from wherever you'd actually bet, but no real money, until you
have **20+ settled bets**.

```bash
# Open a position. Omit --shares for a capped-Kelly suggestion.
python3 -m cs2model.cli bet --capital 500 \
    --market "Vitality vs Spirit" --outcome "Vitality" \
    --prob 0.71 --price 0.58 --bo 3 --event "IEM Cologne"

# Sportsbook odds instead of a Polymarket price:
python3 -m cs2model.cli bet --market "Vitality vs Spirit" --outcome "Vitality" \
    --prob 0.71 --odds 1.75 --stake 50

# When the match finishes:
python3 -m cs2model.cli settle --id a711dca5 --result won

# Or sell out early, in full or in part:
python3 -m cs2model.cli close --id a711dca5 --price 0.82

# Look at everything:
python3 -m cs2model.cli dashboard
```

`--prob` is the model's probability **for the outcome you bought**. If
`predict` says Vitality 71% and you back Spirit, then `--prob` is `0.29`.

### The row that matters

The dashboard's **calibration** line compares what the model predicted against
what actually happened:

- `calibrated` — its confidence is real. This is what you're waiting for.
- `OVER-CONFIDENT` (red) — it claims more certainty than it delivers. Raise
  your threshold or stop.
- `too few bets to judge` — below 20 settled bets it refuses to guess, because
  a verdict off three bets is noise.

Betting real money before this says `calibrated` is betting on an untested claim.

---

## Part 7 — Keep it running

```bash
tmux new -s cs2
python3 -m cs2model.cli run --capital 500 --interval 300
```

`--capital` is required the first time — a ledger created with zero capital
halts immediately on "capital exhausted", which looks exactly like a crash.
The bot is in DRY RUN unless you add `--live`.

Press **Ctrl+B** then **D** to detach. It keeps running with your laptop shut.
Come back with `tmux attach -t cs2`.

In another terminal:

```bash
python3 -m cs2model.cli dashboard --watch 30
```

✅ Top line reads `● ALIVE`. If it reads `✕ STALE`, the loop died — reattach
to tmux and look at the error.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: numpy` | venv not active | `cd polymarket-test && source venv/bin/activate` |
| `command not found: python3` | Python not installed | Part 1.2 |
| `pip install` says `Killed` | Out of RAM | Add swap, Part 1.4 |
| `LIQUIPEDIA_API_KEY is not set` | Env var didn't stick | Part 3.1, then reopen the terminal |
| `HTTP 429` | Rate limited | Wait. Never lower `SLEEP_SECONDS` |
| `Not enough data to evaluate` | Too few matches ingested | `--pages 80`, or lower `--initial-frac` |
| `unknown team 'X'` | Name doesn't match the data | Use a name from the list it prints |
| Dashboard says `✕ STALE` | `run` loop died | `tmux attach -t cs2` and read the error |
| `ingested 0 usable matches` | Field mapping wrong | Send me your `explore --save` output |

---

## What's still missing

1. **Upcoming fixtures** — nothing yet fetches what's playing tomorrow.
   `_to_match()` drops any record without a winner, which is every unplayed
   match. Needs `data/raw_upcoming.json` from step 3.3 to build.
2. **Auto-settle** — the `run` loop currently only heartbeats. It should check
   whether open positions have resolved and settle them.
3. **Odds** — typed by hand. Bookmakers don't give free APIs and scraping them
   generally breaks their terms.

Steps 3.2 and 3.3 are what unblock 1 and 2.
