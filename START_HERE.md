# Start here

This guide assumes you have never done any of this before. Nothing is
skipped. If a word looks like jargon, it is explained in the glossary at the
bottom.

**How to use this guide:** do one step. Check the "You should see" bit. If it
matches, go to the next step. If it does not match, stop and send me exactly
what you saw. Do not skip ahead.

---

## Things you need before you start

| # | What | Do you have it? | Cost | When you need it |
|---|---|---|---|---|
| 1 | Your AWS server | ✅ you said you have one | — | Step 1 |
| 2 | The `.pem` key file for that server | check your Downloads folder | — | Step 1 |
| 3 | A Liquipedia API key | ❌ you need to get this | Free | Step 6 |
| 4 | A Polymarket account | ❌ **not yet** | Free | Much later |
| 5 | Money in a crypto wallet | ❌ **not yet** | — | Much later, maybe never |

**Only #3 matters right now.** Ignore #4 and #5 for now. We are not touching
real money for a long time. I will tell you clearly when that changes.

---

## Step 1 — Open a terminal on your own computer

A **terminal** is a window where you type commands instead of clicking
buttons. It looks like a black or white box with text.

**On Windows:** Click Start. Type the word `powershell`. Press Enter.

**On Mac:** Hold the `Command` key and press the spacebar. Type the word
`terminal`. Press Enter.

**You should see:** a window with some text and a blinking cursor.

That is it. That is the whole step.

---

## Step 2 — Connect to your server

Your server is a computer that lives in a data centre and is always on. You
control it by typing commands into your terminal.

Type this, but replace the two capitalised parts with your own details:

```
ssh -i /path/to/YOUR-KEY.pem ubuntu@YOUR-SERVER-IP
```

- `/path/to/YOUR-KEY.pem` — where the key file is on your computer. If it is
  in your Downloads folder it is probably `~/Downloads/mykey.pem` on Mac, or
  `C:\Users\YourName\Downloads\mykey.pem` on Windows.
- `YOUR-SERVER-IP` — a number like `54.123.45.67`. Find it in the AWS
  website: log in → search "EC2" → click "Instances" → click your server →
  look for **Public IPv4 address**.

Press Enter. The first time it asks `Are you sure you want to continue
connecting?` — type `yes` and press Enter.

**You should see:** the text at the start of the line changes to something
like `ubuntu@ip-172-31-0-1:~$`

**That means you are now typing commands ON THE SERVER, not on your own
computer.** Everything from here happens on the server.

### If it did not work

| What you saw | What it means | What to do |
|---|---|---|
| `Permission denied (publickey)` | Wrong username | Try `ec2-user@` instead of `ubuntu@`, then `admin@` |
| `UNPROTECTED PRIVATE KEY FILE` | Your key file is too open | Run `chmod 400 /path/to/YOUR-KEY.pem` then try again |
| It just sits there, then times out | Your server is blocking you | In AWS: EC2 → Instances → click your server → Security tab → click the security group → Edit inbound rules → Add rule → choose `SSH` → Source `My IP` → Save |
| `No such file or directory` | Wrong path to the key file | Check where the `.pem` file actually is |

---

## Step 3 — Install the tools the program needs

Copy and paste this whole line, then press Enter:

```
sudo apt update && sudo apt install -y python3 python3-venv python3-pip git tmux
```

This installs Python (the language the program is written in) and a few
helpers. It takes a minute or two and prints a lot of text. That is normal.

> If that gives an error saying `apt: command not found`, your server is a
> different flavour of Linux. Use this instead:
> ```
> sudo dnf install -y python3 python3-pip git tmux
> ```

Now check Python is there:

```
python3 --version
```

**You should see:** `Python 3.11.5` or similar. Any number starting with
`3.9` or higher is fine.

---

## Step 4 — Download the program

```
git clone https://github.com/Smileifyouliked/polymarket-test.git
```

```
cd polymarket-test
```

```
git checkout claude/polymarket-github-repos-xomddj
```

What these do, in order: download the code, move into the folder, switch to
the version with all the latest work in it.

```
ls
```

**You should see:** a list including `cs2model`, `tests`, `SETUP.md`,
`START_HERE.md`.

---

## Step 5 — Install the program's parts

```
python3 -m venv venv
```

This makes a private sandbox so this program cannot break anything else on
the server.

```
source venv/bin/activate
```

**You should see:** your line now starts with `(venv)`. Like this:
`(venv) ubuntu@ip-172-31-0-1:~$`

```
pip install -r requirements-cs2.txt
```

This downloads the maths libraries the program uses. Takes 1-3 minutes.

**You should see:** it ends with the words `Successfully installed` followed
by a list of names.

> ### ⚠️ The mistake everyone makes
>
> **Every single time you close your terminal and come back, you MUST type
> these two lines first:**
> ```
> cd polymarket-test
> source venv/bin/activate
> ```
> If you forget, commands fail with `ModuleNotFoundError: No module named
> 'numpy'`. That error always means "you forgot these two lines". Nothing is
> broken. Just type them and carry on.

> ### If `pip install` says `Killed`
>
> Your server ran out of memory. This happens on the smallest AWS servers.
> Run these three lines once, then run `pip install` again:
> ```
> sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
> sudo mkswap /swapfile && sudo swapon /swapfile
> ```

---

## Step 6 — Check the program works

This step needs no key and no account. It uses made-up practice data.

```
python3 -m pytest tests/ -q
```

**You should see:** `79 passed`

That means all 79 internal checks pass. Now the real test:

```
python3 -m cs2model.cli demo
```

It thinks for about 30 seconds, then prints some tables.

**You should see:** a table with the words `COVERAGE CURVE` in it, showing
confidence levels and accuracy percentages.

### 🎉 If you got here, the program works on your server.

Everything so far used pretend data. Now we get real data.

---

## Step 7 — Get your Liquipedia key

**What is this?** Liquipedia is a website that records every professional
Counter-Strike match ever played. We need that history to teach the model who
is good. They give the data away free, but they want you to identify
yourself, so they give you a **key** — a long password-like string that says
"this is me".

**What to do:**

1. Open a normal web browser on your own computer.
2. Go to: **https://liquipedia.net/api-terms-of-use**
3. Follow whatever it asks. It is free. You will likely need to make an
   account and agree to their rules.
4. At the end you get a key. It looks like a long jumble of letters and
   numbers.
5. Copy it.

> I cannot see that website from where I am, so I cannot tell you exactly
> which buttons to press. If any part confuses you, take a screenshot or copy
> the text and send it to me.

**Now put the key on your server.** Back in your terminal, type these three
lines, replacing `PASTE_YOUR_KEY_HERE` with the key you copied, and
`your@email.com` with your real email:

```
echo 'export LIQUIPEDIA_API_KEY=PASTE_YOUR_KEY_HERE' >> ~/.bashrc
```

```
echo 'export LIQUIPEDIA_USER_AGENT="cs2model/0.1 (your@email.com)"' >> ~/.bashrc
```

```
source ~/.bashrc
```

Check it worked:

```
echo $LIQUIPEDIA_API_KEY
```

**You should see:** your key printed back at you.

> **The email is not optional.** Liquipedia asks who is using their data and
> blocks programs that will not say. Use a real email.

> ### 🔒 Keep the key private
> Never put this key in a message, a file inside the `polymarket-test`
> folder, or anywhere on the internet. The `~/.bashrc` file we used is
> outside the project folder on purpose, so it can never be uploaded by
> accident.

---

## Step 8 — The two commands I need from you

This is the step everything else is waiting on.

**Why:** I wrote the part of the program that reads Liquipedia's data by
following their written instructions — but I have never been able to actually
connect to them and see what their data looks like. So my code is an educated
guess. These two commands show me the real thing so I can correct it.

**Command one:**

```
python3 -m cs2model.cli explore --limit 3 --save data/raw_finished.json
```

**Command two:**

```
python3 -m cs2model.cli explore --limit 3 --conditions "[[winner::]]" --save data/raw_upcoming.json
```

The first gets matches that already happened. The second gets matches that
have not been played yet.

**Then send me:**
- Everything the commands printed on screen (copy and paste it), **and**
- The two files they created, which are at `data/raw_finished.json` and
  `data/raw_upcoming.json`

To see a file's contents so you can copy it:

```
cat data/raw_finished.json
```

**These files contain match results only. There is no password or key inside
them. They are safe to share.**

### If you get an error, that is still useful — send it to me

| Error message | What it means |
|---|---|
| `LIQUIPEDIA_API_KEY is not set` | Step 7 did not stick. Close the terminal, connect again, redo Step 7 |
| `HTTP 401` | The key is wrong, or not switched on yet |
| `HTTP 429` | You asked too fast. Wait 10 minutes and try once more |
| `Could not find a working base URL` | My guess at their web address was wrong. Send me the whole message |
| `ModuleNotFoundError` | You forgot `source venv/bin/activate` |

---

## What happens after that

You do not need to do these yet. This is just so you know where we are going.

| Step | What it is | Real money? |
|---|---|---|
| 9 | I fix the code using your files | No |
| 10 | Download years of match history | No |
| 11 | **Test if the model is actually any good** | No |
| 12 | Run the bot in practice mode for 1-2 weeks | No |
| 13 | Only if step 11 and 12 both look good — consider real money | Yes |

**Step 11 is the important one.** It tells us whether this whole thing
predicts matches better than just picking whoever is favourite. If it does
not, we stop and fix it rather than betting on it. I would rather find that
out for free than with your money.

---

## Things NOT to do yet

- ❌ Do not make a Polymarket account yet
- ❌ Do not put any money anywhere
- ❌ Do not use the `--live` option on any command
- ❌ Do not put a crypto wallet key on the server

None of these are needed for anything in this guide. When the time comes I
will explain each one properly first.

---

## Glossary

**Terminal** — the text window where you type commands.

**Command** — a line of text you type and press Enter on. It tells the
computer to do one thing.

**Server / VPS** — a computer in a data centre that is always switched on.
Yours is at AWS.

**SSH** — the way you connect from your computer to your server.

**`.pem` file** — a key file. Like a house key, but for your server. Never
share it.

**Python** — the programming language this project is written in.

**venv (virtual environment)** — a private sandbox for this project's parts,
so it cannot break anything else.

**API key** — a long password that identifies you to a website when your
program talks to it instead of you clicking around.

**Liquipedia** — the website with the history of every pro Counter-Strike
match.

**Polymarket** — the website where you could eventually place bets. Not yet.

**Dry run / practice mode** — the program does everything except spend money.

**The model** — the part that works out who is likely to win.

**tmux** — a tool that keeps a program running on the server after you close
your laptop. You will need it later, not yet.

---

## If you get stuck

Send me:
1. The exact command you typed
2. The exact text that came back (copy and paste all of it)

Do not worry about breaking anything. The sandbox in Step 5 keeps this
project separate from everything else, and nothing in this guide touches
money or any account except a free data website. Errors are completely normal
and are usually a one-line fix.
