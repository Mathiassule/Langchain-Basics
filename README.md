# LangChain Basics: Q&A Assistant

A production-ready, modular Question & Answering (Q&A) Assistant built with **LangChain (LCEL)** and the **OpenAI API**, featuring conversation history, safe environment key management, input validation, and fine-grained error handling.

---

## Deliverables Checklist

This repository satisfies all required module deliverables:

| Deliverable | Location in Repository | Status |
| :--- | :--- | :--- |
| **Python Script** | [qa_assistant.py](qa_assistant.py) | Complete |
| **Python Notebook** | [qa_assistant.ipynb](qa_assistant.ipynb) | Complete |
| **Dependencies Manifest** | [requirements.txt](requirements.txt) | Complete |
| **Environment Variable Template** | [.env.example](.env.example) | Complete (no real API key) |
| **Five Test Questions & Answers** | [test_evidence.md](test_evidence.md) | Complete |
| **Prompt & Chain Explanation** | [explanation.md](explanation.md) | Complete |
| **Errors, Risks, & Limitations** | [errors_risks_limitations.md](errors_risks_limitations.md) | Complete |
| **Setup & Running Instructions** | Below in this README | Complete |

---

## 1. Core Concepts in Simple Product & Engineering Terms

| Concept | Product Perspective | Engineering Perspective |
| :--- | :--- | :--- |
| **Prompt** | The user persona, behavioral guardrails, and task description for the AI. | A parameterized template (`ChatPromptTemplate`) dynamically populating system roles, message history, and user inputs into formatted API payloads. |
| **Model** | The cognitive AI engine (e.g., GPT-4o-mini) generating human-readable responses. | A stateless inference endpoint (`ChatOpenAI`) that receives structured message arrays and outputs response tokens. |
| **Chain** | An end-to-end automated pipeline turning customer questions into refined answers. | Composable pipeline using LangChain Expression Language (`prompt \| model \| StrOutputParser()`). |
| **Memory / History** | The assistant's ability to recall previous conversation turns without user repetition. | A sliding-window message buffer (`List[BaseMessage]`) passed into the prompt's `MessagesPlaceholder`. |
| **Agent** | An autonomous assistant that can execute tools, query databases, and solve multi-step problems. | A runtime reasoning loop where an LLM dynamically selects tools/functions to invoke until completing a goal. |

---

## 2. Project Architecture

```
[User Input]
     │
     ▼
[Input Validation: Empty / Whitespace Check]
     │
     ▼
[History Manager: Sliding Window Trimming (5 turns)]
     │
     ▼
[ChatPromptTemplate] ─── System Instructions + History Placeholder + User Question
     │
     ▼
[ChatOpenAI (gpt-4o-mini)] ─── Temperature: 0.7, Safe API Key (.env)
     │
     ▼
[StrOutputParser] ─── Extracts clean response text
     │
     ▼
[Error Handling Wrapper] ─── Catches AuthenticationError, RateLimitError, APIConnectionError
     │
     ▼
[Record Turn in History] ─── Appends (HumanMessage, AIMessage)
     │
     ▼
[Clean Response to User]
```

---

## 3. Setup and Installation Instructions

### Prerequisites
- Python 3.10, 3.11, or 3.12 installed.
- (Optional) An OpenAI API key from [platform.openai.com](https://platform.openai.com).

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd Langchain-Basics
```

### Step 2: Create and Activate a Virtual Environment
- **Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
- **macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Copy the template file to `.env`:
- **Windows (PowerShell):**
  ```powershell
  Copy-Item .env.example .env
  ```
- **macOS / Linux:**
  ```bash
  cp .env.example .env
  ```

Open `.env` in any text editor and supply your OpenAI API key:
```env
OPENAI_API_KEY=sk-your-actual-api-key-here
OPENAI_MODEL_NAME=gpt-4o-mini
TEMPERATURE=0.7
MAX_HISTORY_TURNS=5
```
> **Security Note:** `.env` is listed in `.gitignore` and must **never** be committed to version control.

---

## 4. How to Run the Application

### Option A: Run the Automated 5-Question Test Suite
Runs the full evaluation suite including the 5 contextual questions and edge-case error tests:

- **With live OpenAI API key:**
  ```bash
  python qa_assistant.py --test
  ```
- **In demo mode (offline / without spending credits):**
  ```bash
  python qa_assistant.py --demo --test
  ```

### Option B: Interactive Chat Mode
Launches a terminal Q&A chat where you can converse with the assistant, preserve history, and inspect memory:
```bash
python qa_assistant.py
```
*Special Commands during chat:*
- `history` - View all messages currently stored in active memory.
- `clear`   - Reset conversation history.
- `exit` or `quit` - Exit the chat.

### Option C: Single Question Query
```bash
python qa_assistant.py --question "What is LangChain in simple terms?"
```

### Option D: Jupyter Notebook
To run the interactive notebook with step-by-step code and rich outputs:
```bash
jupyter notebook qa_assistant.ipynb
```

---

## 5. Summary of the Five Test Questions

The assistant was evaluated with five sequential questions designed to test domain understanding, contextual memory, and production safety:

1. **Question 1 (Domain Understanding):**  
   *"What is LangChain and what problem does it solve for software developers?"*  
   *Result:* Explains composability, context-awareness (RAG), and agency.
2. **Question 2 (Contextual Memory Test):**  
   *"Can you compare the concepts of 'Chains' and 'Agents' that you just mentioned?"*  
   *Result:* Resolves the reference to terms from Q1; compares deterministic workflows vs. dynamic reasoning loops.
3. **Question 3 (Technical Implementation):**  
   *"How does LCEL (LangChain Expression Language) make building these chains easier?"*  
   *Result:* Covers Unix-style pipe syntax (`|`), unified runtime methods, streaming, and production readiness.
4. **Question 4 (Grounded Code Generation):**  
   *"Can you show a minimal Python code snippet illustrating that LCEL pipe pattern?"*  
   *Result:* Returns an executable code snippet combining `prompt | model | parser`.
5. **Question 5 (Production Engineering & Security):**  
   *"What are three critical security and reliability risks when deploying this in production?"*  
   *Result:* Covers credential leakage, prompt injection, and rate limits/runaway token costs.

*Full empirical outputs, token states, and edge-case results are documented in [test_evidence.md](test_evidence.md).*

---

## 6. Prompt and Chain Design

- **Prompt Template:** Built using `ChatPromptTemplate.from_messages()` combining:
  1. A `SystemMessage` establishing a concise, professional persona with instructions to admit uncertainty.
  2. A `MessagesPlaceholder("history")` that dynamically injects active multi-turn context.
  3. A `HumanMessage("{question}")` cleanly isolating user text.
- **LCEL Chain:**
  ```python
  chain = prompt | model | StrOutputParser()
  ```
  This creates a declarative, type-safe data pipeline that streams clean string outputs while handling token conversion automatically.

*Read the complete deep dive in [explanation.md](explanation.md).*

---

## 7. Errors, Risks, and Limitations

- **Input Errors:** Empty and whitespace-only queries (`"   "`) are rejected immediately without consuming API quota.
- **API Errors:** Explicitly captures `openai.AuthenticationError`, `openai.RateLimitError`, and `openai.APIConnectionError` with clear remediation messages instead of raw crash traces.
- **Risks:** Addresses hallucinations, prompt injection, API key exposure, and token cost runaway.
- **Limitations:** Explains in-memory statelessness, context window boundaries, and latency overhead.

*Read the full risk and error analysis in [errors_risks_limitations.md](errors_risks_limitations.md).*

---

## 8. Repository File Structure

```
Langchain-Basics/
├── .env.example                # Safe environment variable configuration template
├── .gitignore                   # Excludes .env, venv, and cache files from Git
├── README.md                    # Project overview, architecture, and setup instructions
├── requirements.txt             # Project dependencies (LangChain, OpenAI, etc.)
├── qa_assistant.py              # Main Q&A assistant Python script & CLI runner
├── qa_assistant.ipynb           # Interactive Jupyter notebook deliverable
├── test_evidence.md             # Documented evidence of 5 test Q&As + edge cases
├── explanation.md               # Detailed prompt, chain, and concepts explanation
└── errors_risks_limitations.md  # Comprehensive guide to errors, risks, and limitations
```
