import os
import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DRIVE_LINKS = {
    "Domestic Relocation Policy India 2023.pdf": "https://drive.google.com/file/d/11k4DQFxa0F9-jkI_nfxxSP5xu6656iIg/view?usp=drive_link",
    "Holiday Calendar 2026 - Bangalore & Rest of India - Sigmoid.pdf": "https://drive.google.com/file/d/1seIp9bDAL50uV9rLwssvKth-pJPcF89H/view?usp=drive_link",
    "Salary Advance Policy- 2025.pdf": "https://drive.google.com/file/d/1g1dE7Rtc7M148oOS0-vuqwD-Dl56m148/view?usp=drive_link",
    "Sigmoid Company Event Participation Policy-2025.pdf": "https://drive.google.com/file/d/1mN26F8x3AiCk1BUTlkcZiXyljyhgc0TP/view?usp=drive_link",
    "Sigmoid-POSH-2026.pdf": "https://drive.google.com/file/d/1fKJyf6s0wbRjXbWm4z7FvgFADNf48odL/view?usp=drive_link",
    "Social Media Policy-Version 1.0.pdf": "https://drive.google.com/file/d/1cJzfY86XyFXckZgNKKqwl0zKXDefz1lt/view?usp=drive_link",
    "Time-Off Policy Ver 5.1.pdf": "https://drive.google.com/file/d/1RKR52NVpYr5JQXc7bmtFp8mKjBHjY2g0/view?usp=drive_link",
    "PROBATION CONFIRMATION POLICY.pdf": "https://drive.google.com/file/d/1FTlobfg-yshcQ1kBnTXHmxgSpDzKedAc/view?usp=drive_link",
    "Reimbursement Policy (India).pdf": "https://drive.google.com/file/d/1y9RnWtICkuaARTqghv34F6vuUhg57IRf/view?usp=drive_link",
    "Working from Remote Locations - Guidelines.pdf": "https://drive.google.com/file/d/1gExPea-SFOSmuyx8Gk9OvDwlpO8J00cA/view?usp=drive_link"
}


def load_documents(folder_path):
    documents = []

    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)
        file_name = os.path.basename(path)

        # 🔹 Handle PDFs
        if file.endswith(".pdf"):
            with pdfplumber.open(path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        documents.append(
                            Document(
                                page_content=text,
                                metadata={
                                    "source": file,
                                    "page": i + 1,
                                    "file_path": DRIVE_LINKS.get(file_name, "")
                                }
                            )
                        )

        # 🔹 Handle TXT / MD
        elif file.endswith(".txt") or file.endswith(".md"):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": file,
                            "file_path": DRIVE_LINKS.get(file_name, "")
                        }
                    )
                )

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)