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
| 3 | A free Kaggle account | ❌ you need this | Free | Step 7 |
| 4 | A Polymarket account | ❌ **not yet** | Free | Much later |
| 5 | Money in a crypto wallet | ❌ **not yet** | — | Much later, maybe never |

**Only #3 matters right now.** Ignore #4 and #5 for now. We are not touching
real money for a long time. I will tell you clearly when that changes.

---

## Step 1 — Know what a terminal is

A **terminal** is a window where you type commands instead of clicking
buttons. It looks like a black or white box with text. You will get one in
the next step.

---

## Step 2 — Connect to your server

Your server is a computer that lives in a data centre and is always on. You
control it by typing commands into it.

There are **two ways** to connect. Pick one.

### Way A — the browser button (easier, no key file needed)

1. Log in to AWS in your web browser.
2. Search for **EC2**, click **Instances**, tick your server.
3. Check it says **Running** and **2/2 checks passed**. If it says
   "Initializing", wait 2 minutes — it is still switching on.
4. Click the **Connect** button at the top.
5. Choose the **EC2 Instance Connect** tab, then click **Connect**.

A black terminal window opens inside your browser. That is your server.

> #### If you see "Failed to connect to your instance"
>
> This almost always means your security group is not letting AWS in. The
> browser button connects **from an AWS server, not from your computer**, so a
> rule that says "My IP" will block it.
>
> Fix it like this:
> 1. EC2 → **Instances** → tick your server → **Security** tab
> 2. Click the blue security group link (starts with `sg-`)
> 3. **Inbound rules** → **Edit inbound rules** → **Add rule**
> 4. Type: **SSH**
> 5. Source: choose **Custom**, and in the box type the address for your
>    region:
>
>    | Your region (shown top-right in AWS) | Type this in the box |
>    |---|---|
>    | Europe (Ireland) `eu-west-1` | `18.202.216.48/29` |
>    | Europe (Frankfurt) `eu-central-1` | `3.120.181.40/29` |
>    | Europe (Stockholm) `eu-north-1` | `13.48.4.200/30` |
>
>    Other region? Type `com.amazonaws.` then your region then
>    `.ec2-instance-connect` and pick the entry that appears.
> 6. **Save rules**, wait 30 seconds, try Connect again.
>
> Leave your existing "My IP" rule alone — it does no harm and you need it for
> Way B.

### Way B — from your own terminal (needs the `.pem` key file)

Open a terminal on your own computer:

- **Windows:** Start → type `powershell` → Enter
- **Mac:** hold `Command`, press spacebar → type `terminal` → Enter

Then type this, replacing the two capitalised parts with your own details:

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

What these do: download the code, then move into the folder. The `main`
branch already has everything, so there is no branch to switch to.

> **Already downloaded it earlier?** Get the newest version instead:
> ```
> cd polymarket-test && git checkout main && git pull
> ```
> You do NOT need to reinstall anything. Steps 3 and 5 stay done.

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

## Step 7 — Get the match data

**What changed and why:** the original plan was Liquipedia. It turned out their
free data is only for education and non-commercial sites, and they currently
only sell an Enterprise plan. So we switched to a public dataset of real CS2
matches instead. It is free, it needs no approval, and it is actually the
better test — it covers about 18 months of CS2 rather than a live trickle.

**Nothing you have already done is wasted.** Steps 1 to 6 are unchanged.

### 7.1 Make a Kaggle account

Kaggle is a free website that hosts public datasets. Go to
**https://www.kaggle.com** and sign up.

### 7.2 Get your Kaggle token

1. Click your profile picture (top right) → **Settings**
2. Scroll to **API** → click **Create New Token**
3. A small file called `kaggle.json` downloads
4. Open it in Notepad or TextEdit and copy everything inside. It is one short
   line that looks like `{"username":"...","key":"..."}`

### 7.3 Put the token on your server

```
pip install kaggle
```

```
mkdir -p ~/.kaggle
```

```
nano ~/.kaggle/kaggle.json
```

A simple text editor opens. **Paste** what you copied. Then press `Ctrl+O`,
press `Enter`, then press `Ctrl+X` to save and quit.

```
chmod 600 ~/.kaggle/kaggle.json
```

That last line makes the file private. Kaggle refuses to run without it.

> 🔒 This token is a password. Never put it in a message, or in the
> `polymarket-test` folder. `~/.kaggle/` is outside the project on purpose.

### 7.4 Download the data

```
kaggle datasets download -d griffindesroches/cs2-hltv-professional-match-statistics-dataset
```

```
unzip -o *.zip -d data/kaggle
```

```
ls data/kaggle
```

**You should see:** one or more `.csv` files listed.

---

## Step 8 — The one thing I need from you

```
head -1 data/kaggle/*.csv
```

That prints the first line of the file — the column names.

**Copy that and send it to me.**

**Why:** the file has around 116 columns and I have never seen it. The loader
I wrote expects one row per map; this file is probably one row per match. I
need to see the real column names to make them fit. I would rather look than
guess — guessing at data formats is what caused most of the bugs we have
already had to fix.

Then I will update the loader, and you run these two:

```
python3 -m cs2model.cli load-csv --csv data/kaggle/YOUR_FILE.csv --out data/matches.json
```

```
python3 -m cs2model.cli evaluate --data data/matches.json
```

### 🎯 That second command is the whole point

It prints a table like this:

```
always-pick-favourite   acc= 0.591
FULL MODEL              acc= 0.595
```

If **FULL MODEL** beats **always-pick-favourite** by a clear margin, the model
is worth something and we carry on. If it does not, we stop and fix it rather
than betting on it.

I already ran this on a smaller, older dataset: the model scored 59.5% against
a 59.1% baseline. That is not good enough to bet with. That test used only 3
months of 2015 data though, which is too thin to be fair — this Kaggle set is
the proper test.

### If something goes wrong

| Error | What it means |
|---|---|
| `kaggle: command not found` | You forgot `source venv/bin/activate` |
| `Could not find kaggle.json` | Step 7.3 did not save. Try `cat ~/.kaggle/kaggle.json` to check |
| `403 Forbidden` | The token is wrong, or you have not accepted the dataset's terms — open the dataset page in your browser once and click Download |
| `unzip: command not found` | Run `sudo apt install -y unzip` |
| `could not identify columns` | Expected — that is why I need the header line first |

---

## What happens after that

| Step | What it is | Real money? |
|---|---|---|
| 9 | I adapt the loader to the real columns | No |
| 10 | **Test if the model actually beats picking the favourite** | No |
| 11 | If it passes: connect live Polymarket prices | No |
| 12 | Run the bot in practice mode for 1-2 weeks | No |
| 13 | Only if 10 and 12 both look good — consider real money | Yes |

**Step 10 is the important one.** If the model cannot beat picking the
favourite, nothing after it matters. I would rather find that out for free
than with your money.

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

**Kaggle** — a free website hosting public datasets. This is where the match
history comes from.

**Liquipedia** — another Counter-Strike data source. We do not use it: their
free tier is education-only and they currently sell only an Enterprise plan.

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
