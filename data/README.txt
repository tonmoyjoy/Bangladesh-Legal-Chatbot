Put the Bangladesh law PDFs in this folder, then run:  python ingest.py

PDFs are NOT committed to git (see .gitignore). Official sources (Laws of Bangladesh,
Legislative and Parliamentary Affairs Division) — open each act's page and use
the browser's "Print → Save as PDF":

  1. Constitution of the People's Republic of Bangladesh   https://bdlaws.minlaw.gov.bd/act-367.html
  2. The Penal Code, 1860                                   https://bdlaws.minlaw.gov.bd/act-11.html
  3. The Code of Criminal Procedure, 1898                   https://bdlaws.minlaw.gov.bd/act-75.html
  4. The Muslim Family Laws Ordinance, 1961                 https://bdlaws.minlaw.gov.bd/act-305.html

Scanned (image-only) PDFs are OCR'd automatically when Tesseract is installed
(see README → "Install Tesseract"). Text PDFs need no OCR.

File names are used to label the act (e.g. "Penal code.pdf" → "Penal Code, 1860").
Any other PDF is labelled from its file name — add a pattern in
legalbot/chunker.py (_KNOWN_ACTS) for a nicer name.

To add a new act (e.g. the Acquisition and Requisition of Immovable Property
Act, 2017), drop its PDF here and run `python ingest.py` again — only new or
changed chunks are embedded.
