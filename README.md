# discord-arxiv

A Discord bot that posts each day's new arXiv papers (cs.CV and cs.GR by default)
that are similar to a list of seed papers the server cares about. It stays silent
on days with no relevant papers, and arXiv announces nothing on Friday and Saturday
nights (New York time), so there are no posts on weekends.

Papers come from arXiv's daily announcement feed (`rss.arxiv.org`). New papers and
cross-lists are scored; updated versions of older papers are not. Each paper is
embedded with [SPECTER2](https://huggingface.co/allenai/specter2) and scored by its
cosine similarity to the closest seed paper. Two cutoffs split the results:

- **Relevant** (score ≥ `relevant`): posted in the channel. Each paper shows its
  title as a link, its authors (up to 10, then "+N more"), and its closest seed and
  score. Lists too long for one message continue in further messages.
- **Probably not relevant** (`borderline` ≤ score < `relevant`): a compact list in
  a thread on that message, showing what sits just below the line so the cutoffs
  are easy to tune without flooding the channel.

A SQLite file
records which papers have been processed, so restarts and the hourly retries never
score or post a paper twice.

Everything runs on CPU. Scoring a day's feed (~190 papers) took 46 s on 4 threads
of a desktop Ryzen with a 1.4 GB memory peak; expect a minute or two on a NAS-class
CPU such as an Intel N100.

## Setup

1. **Create the bot.** At <https://discord.com/developers/applications>, create an
   application, open *Bot*, and copy the token. No privileged intents are needed.
   To stop others adding it to their servers, set *Installation → Install Link* to
   *None*, then turn off *Bot → Public Bot* (Discord refuses the second step
   while an install link is set).
2. **Invite it** (replace `APP_ID` with the application ID). This grants View
   Channel, Send Messages, Embed Links, Create Public Threads and Send Messages in
   Threads, plus slash commands:
   `https://discord.com/oauth2/authorize?client_id=APP_ID&scope=bot+applications.commands&permissions=309237664768`
   If the bot is already in the server, open the link again; re-authorizing adds
   the missing permissions and commands without removing anything.
3. **Make the channel read-only.** In the channel's permissions, deny *Send
   Messages* and *Send Messages in Threads* for `@everyone`. Allow the bot *View
   Channel*, *Send Messages*, *Embed Links*, *Create Public Threads* and *Send
   Messages in Threads* (a private channel needs these set on the channel itself).
   Without the thread permissions the bot posts the probably-not-relevant list in
   the channel instead.
4. **Configure.** Copy `.env.example` to `.env` and fill in the token and the
   channel ID (enable Developer Mode in Discord, then right-click the channel →
   *Copy Channel ID*). Edit `config/config.toml` for cutoffs, post time and
   timezone, and the starting seed list.
5. **Run** on the server:

   ```sh
   docker compose up -d --build
   docker compose logs -f
   ```

   The first run downloads the SPECTER2 model (~450 MB) into `data/huggingface`.
6. **Check it works.** Run `/digest` in any channel: after a minute or two it
   shows you today's post exactly as the bot would make it. Only you see it, and
   nothing is posted or marked as seen, so you can run it as often as you like
   (also handy while tuning the cutoffs).

## Seeds

The bot only finds papers similar to at least one seed, so add a few for every
interest the server has. On the first run it imports the `[seeds]` list from
`config/config.toml` into its database; after that, anyone in the server manages
the list with slash commands (in any channel the bot can see):

| Command | What it does |
|---|---|
| `/seeds list` | Shows every seed with its name and title (only you see the reply). |
| `/seeds add paper:<ID or link> [name:<short name>]` | Looks the paper up on arXiv and adds it. The name is shown next to matching papers; it defaults to the title up to its colon ("EDGS: Eliminating…" → "EDGS"). |
| `/seeds remove name:<name>` | Removes a seed. The name autocompletes. |
| `/seeds score paper:<ID or link>` | Scores a paper against the seeds (only you see the reply): its score, its closest seeds, whether the daily post would include it, and what adding it as a seed would change. The first use loads the model, which can take half a minute on a small CPU; it stays loaded for 10 minutes. |

New seeds count from the next daily post. To limit who can add or remove seeds,
use *Server Settings → Integrations → (the bot)* in Discord.

## Cutoffs

Cutoffs are similarities between 0 and 1, set in `config/config.toml` (re-read
before every run, so edits apply without a restart). In practice scores fall
between about 0.80 and 0.97, and a 0.005 change is noticeable. Adding seeds raises
the scores of papers near them, so volume grows with the seed list.

To see what the current settings would post today, run `/digest` in Discord, or
on the command line (inside the container: `docker compose exec arxiv-bot arxiv-bot dry-run`):

```sh
uv run arxiv-bot dry-run
```

The command line version also lists the next few papers below the borderline cutoff.

## Development

```sh
uv sync
uv run pytest
uv run --env-file .env arxiv-bot run   # run the bot locally
```
