# Bangladesh Legal Chatbot — Step-by-Step Setup

Follow these steps **exactly in order** in your VSCode terminal.

---

## STEP 1 — Navigate to Project Folder

```powershell
cd D:\All projects\BD Law\Bangladesh-Legal-Chatbot
```

---

## STEP 2 — Create Virtual Environment

```powershell
python -m venv .venv
```

Wait ~30 seconds. A `.venv` folder will appear in your project.

---

## STEP 3 — Activate Virtual Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

You should see `(.venv)` at the start of your terminal line.

> ✅ **Execution policy has already been fixed.** If you still get an error, run:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
> Type `Y` and press Enter. Then run the activate command again.

---

## STEP 4 — Upgrade pip

```powershell
python -m pip install --upgrade pip
```

---

## STEP 5 — Install All Dependencies

```powershell
pip install -r requirements.txt
```

Wait 3-5 minutes. You will see `Successfully installed...` when done.

---

## STEP 6 — Install OCR Python Packages

```powershell
pip install pytesseract Pillow -U
```

---

## STEP 7 — Install Tesseract Binary

Tesseract is a system program needed for OCR on scanned PDFs.

**Recommended — Winget in VS Code PowerShell:**
```powershell
winget install --id tesseract-ocr.tesseract --exact --accept-package-agreements --accept-source-agreements
```

**Optional — Chocolatey only if it is already installed:**
```powershell
choco install tesseract -y
```

**Fallback — Manual install:**

1. Download the installer from this link:
   https://github.com/UB-Mannheim/tesseract/wiki

2. Run the `.exe` file and click Next → Next → Install

3. Add Tesseract to PATH manually:
   ```powershell
   [Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\Program Files\Tesseract-OCR", "User")
   ```

---

## STEP 8 — Restart VSCode Completely

Close VSCode and reopen it so the terminal picks up the updated PATH.

Then come back to the terminal and run:

```powershell
cd D:\All projects\BD Law\Bangladesh-Legal-Chatbot
```

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## STEP 9 — Verify Tesseract is Working

```powershell
tesseract --version
```

If that says the command is not found, continue anyway if you installed Tesseract in the default location.
The Python OCR loader will also check:
`C:\Program Files\Tesseract-OCR\tesseract.exe`

Expected output:
```
tesseract v5.5.0.20241111
```

```powershell
python -c "import shutil; print('tesseract in PATH:', shutil.which('tesseract'))"
```

Expected output:
```
tesseract in PATH: C:\Program Files\Tesseract-OCR\tesseract.exe
```

```powershell
python -c "import pytesseract; print(pytesseract.get_tesseract_version())"
```

Expected output:
```
5.5.0
```

If any of these fail, go back to STEP 7.

---

## STEP 10 — Create the .env File

```powershell
cp .env.example .env.local
```

---

## STEP 11 — Add Your Groq API Key

Open `.env` in VSCode:

```powershell
code .env
```

Find this line:
```
GROQ_API_KEY=gsk_your_groq_api_key_here
```

Replace it with your real key (get one free at https://console.groq.com):
```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxx
```

Save the file with `Ctrl + S`.

---

## STEP 12 — Add Bangladesh Law PDFs

Download PDFs from http://bdlaws.minlaw.gov.bd and place them in the `data/` folder.

Check the folder has PDFs:
```powershell
dir data\
```

You should see at least one `.pdf` file listed.

---

## STEP 13 — Run Setup Verification

```powershell
python test_setup.py
```

Expected output:
```
✅ Python 3.14.x
✅ fitz (PyMuPDF)
✅ langchain
✅ langchain_community
✅ sentence_transformers
✅ chromadb
✅ streamlit
✅ groq
✅ GROQ_API_KEY set
✅ PDFs found: N file(s)

🎉 All checks passed! You're ready to go.
```

Fix any ❌ errors before moving to the next step.

---

## STEP 14 — Build Vector Store

```powershell
python ingest.py
```

This reads your PDFs, runs OCR if needed, creates embeddings and saves everything to ChromaDB.

Expected output when done:
```
✅ Ingestion complete in XXXs — XXXX chunks indexed.
   Now run:  streamlit run app.py
```

> Scanned PDFs take longer due to OCR — 5 to 50 minutes is normal.

---

## STEP 15 — Launch the Chatbot

```powershell
streamlit run app.py
```

Your browser will open automatically at:
```
http://localhost:8501
```

---

## STEP 16 — Test It

In the sidebar paste your Groq API key if it shows invalid.
Then type a question:

```
What are the fundamental rights guaranteed under the Bangladesh Constitution?
```

You should get an answer based on your PDFs.

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| Execution policy error | `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| Invalid API Key | Get real key from https://console.groq.com and paste in sidebar |
| Tesseract not found | Restart VSCode after installing Tesseract |
| No vector store found | Run `python ingest.py` again |
| ModuleNotFoundError | Run `deactivate` then `.\.venv\Scripts\Activate.ps1` then try again |

---

## Quick Reference (Daily Use)

Every time you want to start the chatbot:

```powershell
cd D:\All projects\BD Law\Bangladesh-Legal-Chatbot
```

```powershell
.\.venv\Scripts\Activate.ps1
```

```powershell
streamlit run app.py
```

---

Done! 🇧🇩⚖️