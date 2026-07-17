# Tax Chat

Tax Chat is a chat assistant that answers questions about the US Tax Code (Title 26 / Internal Revenue Code) for Luminus Analytics. Everything runs on your own computer — no data is sent to any external service.

This guide assumes no prior technical experience. Follow the steps in order.

## Step 1: Install Python

Tax Chat requires Python 3.10 or newer.

1. Go to [python.org/downloads](https://www.python.org/downloads/) and download the latest version.
2. Run the installer. **On Windows, make sure to check the box that says "Add Python to PATH"** before clicking Install.
3. To confirm it worked, open a terminal (**Terminal** on Mac, **Command Prompt** or **PowerShell** on Windows) and type:
   ```bash
   python3 --version
   ```
   You should see something like `Python 3.11.5`. (On Windows, you may need to use `python` instead of `python3`.)

## Step 2: Install Ollama

Ollama runs the AI models on your computer.

1. Go to [ollama.com/download](https://ollama.com/download) and download the installer for your operating system.
2. Run the installer and follow the prompts. Ollama will start running automatically in the background.
3. In your terminal, download the two models Tax Chat needs (this may take a few minutes):
   ```bash
   ollama pull gemma
   ollama pull nomic-embed-text
   ```

## Step 3: Download this project

If you have `git` installed:
```bash
git clone <repository-url>
cd luminus-taxchat
```

Otherwise, click the green **Code** button on the GitHub page, choose **Download ZIP**, then unzip it and open a terminal inside the unzipped folder.

## Step 4: Install the project's dependencies

In your terminal, from inside the project folder, run:
```bash
pip3 install -r requirements.txt
```

Then build the custom embedding model used for search (this uses the `Modelfile` included in the project):
```bash
ollama create nomic-custom -f Modelfile
```

## Step 5: Download the tax code

The raw tax code text isn't included in this project — you need to download it once:

1. Go to [uscode.house.gov/download/download.shtml](https://uscode.house.gov/download/download.shtml)
2. Find **Title 26 - Internal Revenue Code** in the list and download it in **XML** format.
3. Rename the downloaded file to `usc26.xml`.
4. Move it into the `data` folder inside the project (create the `data` folder if it doesn't already exist).

When done, you should have a file at: `luminus-taxchat/data/usc26.xml`

## Step 6: Build the search index

This step processes the tax code file and prepares it for searching. It only needs to be done once, and may take several minutes.

```bash
python3 ingest.py
python3 rag.py build
```

## Step 7: Start the app

```bash
python3 app.py
```

Once you see `Starting Flask server...` in the terminal, open your web browser and go to:

```
http://localhost:5001
```

You can now type tax questions into the chat box and get answers sourced directly from the tax code.

To stop the app, go back to the terminal window and press `Ctrl+C`.

## Running it again later

Once setup is complete, you don't need to repeat Steps 1–6. Just open a terminal in the project folder and run:
```bash
python3 app.py
```
(Make sure Ollama is running first — it usually starts automatically when your computer starts.)

## Troubleshooting

- **"command not found: python3" or "pip3"** — Python wasn't installed correctly, or you need to use `python` / `pip` instead (common on Windows).
- **"Connection refused" or errors mentioning `localhost:11434`** — Ollama isn't running. Open the Ollama application and try again.
- **The app can't find `usc26.xml`** — Double check the file is at `data/usc26.xml` (exact name, lowercase) and that Step 6 was run after placing it there.
- **Port 5001 is already in use** — Run `python3 app.py --port 5002` instead, and open `http://localhost:5002` in your browser.

## How the code works

Tax Chat works in two phases: a one-time **prep phase** that turns the tax code into something searchable, and a **chat phase** that runs every time someone asks a question.

### Phase 1: Preparing the tax code (`ingest.py`, `chunk.py`, `xml_parser.py`)

The tax code is one enormous XML file — too big and too unstructured to hand to an AI model directly. This phase breaks it into small, well-organized pieces:

1. **`xml_parser.py`** reads the raw XML and walks through its structure (titles, subtitles, chapters, sections), pulling out the text and remembering how each piece fits into the hierarchy — for example, that Section 162 belongs to Chapter 1 of Subtitle A.
2. **`ingest.py`** cleans up that text (removing redundant headings, extra whitespace, etc.) and figures out which other sections each piece of text refers to (e.g., "as defined in section 212").
3. **`chunk.py`** splits long sections into bite-sized "chunks" of a few hundred words each — small enough for the AI to process, but not so small that they lose meaning. It tries to split at natural boundaries like `(a)`, `(b)`, `(c)` rather than mid-sentence.

The result is a file, `data/rag_chunks2.json`, containing thousands of these chunks, each tagged with its section number, heading, and where it sits in the tax code's structure.

### Phase 2: Building the search index (`rag.py`)

Before the app can answer questions, it needs a fast way to find the *right* chunks for a given question out of the thousands available. `rag.py` builds two search systems side by side:

- A **keyword search** (like Ctrl+F, but smarter) that's good at matching exact terms and section numbers.
- A **semantic search** that converts each chunk's meaning into a list of numbers (an "embedding") so the computer can find chunks that are conceptually similar to a question, even if they don't share exact words.

Both searches run every time someone asks a question, and their results are blended together — this combination tends to find better matches than either search alone.

### Phase 3: Answering a question (`query.py`)

This is what happens, step by step, when someone types a question into the chat:

1. **Rephrase follow-ups.** If this is a follow-up to earlier messages ("what about for a small business?"), the AI rewrites it into a standalone question first, using the chat history for context.
2. **Classify the question.** The AI decides what *kind* of question it is — for example, is the person asking for a specific calculation, a general list of options, or the definition of a term? This decision controls how thorough the next steps will be.
3. **Look up any sections named directly.** If the question mentions a specific section number, that section is fetched immediately, guaranteeing it's included.
4. **Expand the question.** The AI generates a handful of related search queries to cast a wider net (e.g., one question might become three or four searches covering different angles).
5. **Search the tax code.** Each of those searches runs against the index built in Phase 2, and the results are combined and trimmed down to the most relevant chunks.
6. **Pull in cross-references.** If the top chunks mention other sections (e.g., "except as provided in section 401"), a few of those referenced sections are pulled in too, so the answer doesn't miss important exceptions.
7. **Write the answer.** All the selected tax code text is handed to the AI along with the original question, and it writes an answer that cites its sources.
8. **Double-check the answer.** A second pass asks the AI to verify its own answer against the source text. If something looks off, the answer is rewritten once before being shown to the user.

### The web page (`app.py`, `templates/index.html`)

This is the simplest part: a small Flask web server shows the chat window in your browser, sends each question you type to the steps above, and displays the answer that comes back. It also remembers the conversation on your screen so follow-up questions have context.
