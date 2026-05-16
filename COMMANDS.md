# Bangladesh Legal Chatbot — VS Code Terminal Commands

Run these commands in the VS Code integrated terminal only.

Open it with Terminal > New Terminal, then make sure the shell prompt shows PowerShell (`PS`).

---

## One-Time Setup

### 1. Go to the project folder
```powershell
cd D:\All projects\BD Law\Bangladesh-Legal-Chatbot
```

### 2. Create the virtual environment
```powershell
python -m venv .venv
```

### 3. Activate the virtual environment
```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Upgrade pip
```powershell
python -m pip install --upgrade pip
```

### 5. Install project dependencies
```powershell
pip install -r requirements.txt
```

### 6. Install OCR Python packages
```powershell
pip install pytesseract Pillow -U
```

### 7. Install Tesseract OCR

Recommended in the VS Code terminal:
```powershell
winget install --id tesseract-ocr.tesseract --exact --accept-package-agreements --accept-source-agreements
```

If Winget is not available, use the installer from:
https://github.com/UB-Mannheim/tesseract/wiki

If you install manually, add Tesseract to your user PATH:
```powershell
[Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\Program Files\Tesseract-OCR", "User")
```

### 8. Restart VS Code

Close VS Code completely and reopen it so the terminal picks up the updated PATH.

Then run:
```powershell
cd D:\All projects\BD Law\Bangladesh-Legal-Chatbot
```

```powershell
.\.venv\Scripts\Activate.ps1
```

### 9. Verify Tesseract
```powershell
tesseract --version
```

If that command is not found, check the install location from Python:
```powershell
python -c "import shutil; print(shutil.which('tesseract'))"
```

Python can also use the default Windows install folder directly:
`C:\Program Files\Tesseract-OCR\tesseract.exe`

```powershell
python -c "import pytesseract; print(pytesseract.get_tesseract_version())"
```

### 10. Create the `.env` file
```powershell
cp .env.example .env
```

### 11. Edit `.env`
```powershell
code .env
```

Replace this line:
```text
GROQ_API_KEY=gsk_your_groq_api_key_here
```

With your real API key from https://console.groq.com.

### 12. Add your PDFs

Put Bangladesh law PDFs in the `data/` folder.

Check the files:
```powershell
dir data\
```

### 13. Verify the setup
```powershell
python test_setup.py
```

### 14. Build the vector store
```powershell
python ingest.py
```

---

## Daily Use

Run these three commands in the VS Code terminal whenever you want to start the chatbot:

```powershell
cd D:\All projects\BD Law\Bangladesh-Legal-Chatbot
```

```powershell
.\.venv\Scripts\Activate.ps1
```

```powershell
streamlit run app.py
```

The app opens at http://localhost:8501.

---

## If Something Fails

If activation shows an execution policy error:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

If Tesseract is still not found, restart VS Code after installing it and run the verification commands again.
