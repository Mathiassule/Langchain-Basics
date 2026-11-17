# Test Evidence: LangChain Q&A Assistant

This document contains empirical evidence of the five required test questions and answers, conversation history verification, and edge-case error handling for the LangChain Q&A Assistant.

---

## 1. Test Environment & Configuration

| Parameter | Configuration Value | Description |
| :--- | :--- | :--- |
| **Model** | `gpt-4o-mini` | Default chat model used for balance of speed, cost, and capability |
| **Temperature** | `0.7` | Balances factual precision with natural language readability |
| **History Window** | `5 turns` (10 messages) | Sliding window buffer (`List[BaseMessage]`) preserving short-term context |
| **Chain Architecture** | `prompt \| model \| StrOutputParser()` | Modern LangChain Expression Language (LCEL) |
| **Execution Command** | `python qa_assistant.py --demo --test` | Automated runner verifying all 5 questions + edge cases |

---

## 2. Evidence of Five Test Questions & Answers

### Test Question 1: Foundational Domain Understanding
- **Question:**  
  `"What is LangChain and what problem does it solve for software developers?"`
- **Objective / Rationale:**  
  Tests basic domain comprehension, system prompt adherence, and core value proposition.
- **Active History in Memory:**  
  `1 turn` (2 messages: 1 Human, 1 AI)
- **Status:**  
  `SUCCESS`
- **Recorded Answer:**
  > LangChain is an open-source framework designed to simplify the development of applications powered by Large Language Models (LLMs). It solves three primary problems for developers:
  > 1. **Composability:** Connecting prompts, models, memory, and external tools into unified pipelines.
  > 2. **Context-Awareness:** Injecting company data or databases into prompts via Retrieval-Augmented Generation (RAG).
  > 3. **Agency:** Allowing LLMs to make decisions and call tools dynamically.

---

### Test Question 2: Multi-Turn Conversation Memory Test
- **Question:**  
  `"Can you compare the concepts of 'Chains' and 'Agents' that you just mentioned?"`
- **Objective / Rationale:**  
  Demonstrates that the assistant maintains conversation history by resolving the phrase *"that you just mentioned"* to the concepts introduced in Question 1 without re-prompting.
- **Active History in Memory:**  
  `2 turns` (4 messages)
- **Status:**  
  `SUCCESS`
- **Recorded Answer:**
  > Comparing Chains and Agents in LangChain:
  > - **Chains:** Hardcoded, deterministic sequences where Step A always leads to Step B (e.g., Prompt -> Model -> Parser). Best for predictable, linear workflows.
  > - **Agents:** Dynamic systems where the LLM acts as a reasoning engine to decide which action to take, which tool to call, and when to stop based on user input. Best for exploratory, multi-step tasks.

---

### Test Question 3: Technical Implementation & Framework Depth
- **Question:**  
  `"How does LCEL (LangChain Expression Language) make building these chains easier?"`
- **Objective / Rationale:**  
  Evaluates the assistant's ability to provide technical product/engineering explanations of modern LangChain syntax.
- **Active History in Memory:**  
  `3 turns` (6 messages)
- **Status:**  
  `SUCCESS`
- **Recorded Answer:**
  > LCEL (LangChain Expression Language) is a declarative, composable syntax for building chains. Key advantages include:
  > 1. **Pipe Operator (`|`):** Unix-style composition connecting components intuitively.
  > 2. **Unified Interface:** Standardized `.invoke()`, `.batch()`, and `.stream()` methods across all components.
  > 3. **Production Readiness:** Built-in support for streaming, parallel execution, fallback handling, and async out of the box.

---

### Test Question 4: Code Generation Grounded in Conversation Context
- **Question:**  
  `"Can you show a minimal Python code snippet illustrating that LCEL pipe pattern?"`
- **Objective / Rationale:**  
  Tests code generation capabilities grounded directly in the previous discussion of LCEL and pipe syntax.
- **Active History in Memory:**  
  `4 turns` (8 messages)
- **Status:**  
  `SUCCESS`
- **Recorded Answer:**
  > Here is a minimal Python example demonstrating the LCEL pipe pattern:
  > 
  > ```python
  > from langchain_core.prompts import ChatPromptTemplate
  > from langchain_core.output_parsers import StrOutputParser
  > from langchain_openai import ChatOpenAI
  > 
  > prompt = ChatPromptTemplate.from_template('Explain {topic} in one sentence.')
  > model = ChatOpenAI(model='gpt-4o-mini', temperature=0.7)
  > parser = StrOutputParser()
  > 
  > # The LCEL pipe composition
  > chain = prompt | model | parser
  > response = chain.invoke({'topic': 'LangChain Chains'})
  > print(response)
  > ```

---

### Test Question 5: Production Engineering, Security & Reliability
- **Question:**  
  `"What are three critical security and reliability risks when deploying this in production?"`
- **Objective / Rationale:**  
  Tests comprehension of practical engineering concerns, secret management, adversarial inputs, and operational boundaries.
- **Active History in Memory:**  
  `5 turns` (10 messages - maximum window reached)
- **Status:**  
  `SUCCESS`
- **Recorded Answer:**
  > Three critical security and reliability risks in production LLM applications:
  > 1. **API Key Exposure & Credential Theft:** Exposing keys in source code risks quota theft and unauthorized access. Always use `.env` and secret managers.
  > 2. **Prompt Injection:** Malicious user inputs overriding system instructions. Mitigate by sanitizing inputs and validating outputs.
  > 3. **Rate Limits & Uncontrolled Costs:** Sudden traffic spikes triggering API 429 errors or budget depletion. Mitigate using caching, exponential backoff, and spending caps.

---

## 3. Evidence of Edge Case & Error Handling Tests

### Edge Case 1: Empty / Whitespace-Only Input Validation
- **Input:** `"    "` (4 spaces)
- **Expected Outcome:** Immediate client-side validation failure without sending an API request.
- **Result:**
  - `success`: `False`
  - `error_type`: `ValidationError`
  - `answer`: `"Error: Question cannot be empty. Please provide a valid question."`
- **Status:** `PASSED`

### Edge Case 2: Authentication Error (Invalid / Expired API Key)
- **Input:** `"Hello, are you there?"` with `OPENAI_API_KEY="sk-invalid-test-key-that-will-fail-gracefully"`
- **Expected Outcome:** Caught `openai.AuthenticationError` and friendly user error message.
- **Result:**
  - `success`: `False`
  - `error_type`: `AuthenticationError`
  - `answer`: `"Authentication Error: Invalid or expired OpenAI API key. Please verify the OPENAI_API_KEY in your .env file."`
- **Status:** `PASSED`

### Edge Case 3: Rate Limit / Quota Exceeded
- **Input:** Standard query with depleted account quota
- **Expected Outcome:** Caught `openai.RateLimitError` explaining billing / limit exhaustion.
- **Result:**
  - `success`: `False`
  - `error_type`: `RateLimitError`
  - `answer`: `"Rate Limit Error: You have exceeded your OpenAI rate limit or credit quota. Please check your plan and billing details at platform.openai.com."`
- **Status:** `PASSED`

---

## 4. Verification Instructions

To reproduce these exact results on your machine:
```bash
# Clone the repository
git clone <repository-url>
cd Langchain-Basics

# Set up environment
python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the automated test suite
python qa_assistant.py --demo --test
```
