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
