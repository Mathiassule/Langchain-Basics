# LangChain Architecture & Concept Explanation

This document explains the core concepts of LangChain (**prompts**, **models**, **chains**, and **agents**) in both product and engineering terms, followed by a breakdown of the specific prompt template and chain implemented in this project.

---

## 1. Core Concepts: Product vs. Engineering Perspective

| Concept | What It Means to a Product Manager | What It Means to a Software Engineer |
| :--- | :--- | :--- |
| **Prompt** | The specification and persona given to an AI feature to ensure it speaks with the brand voice, stays on topic, and delivers the intended user experience. | A parameterized template (e.g., `ChatPromptTemplate`) that dynamically formats strings and role-based messages (`SystemMessage`, `HumanMessage`, `AIMessage`) into the payload required by LLM APIs. |
| **Model** | The "brain" or engine powering the feature (e.g., GPT-4o-mini). Faster models reduce user drop-off; smarter models handle complex reasoning. | A stateless, remote API client (e.g., `ChatOpenAI`) that accepts an array of tokenized messages and produces a probability distribution over subsequent completion tokens. |
| **Chain** | An end-to-end automated workflow that transforms customer input into a clean, final product response. | A composable data pipeline (using LCEL: `prompt \| model \| parser`) where components implement a standard `Runnable` interface supporting `.invoke()`, `.stream()`, and `.batch()`. |
| **Memory / History** | The assistant's ability to remember what the user said earlier in the conversation so the user doesn't have to repeat themselves. | An ephemeral or persistent state array passed into the model prompt on each turn, often trimmed with a sliding window or summarizer to stay within token context limits. |
| **Agent** | A proactive assistant that can research information, check databases, or take actions on behalf of the customer rather than just answering questions. | A dynamic decision-making loop where an LLM is equipped with callable tools/functions and iteratively determines which tool to execute based on runtime outputs until a stop condition is met. |

---

## 2. Explanation of the Prompt Used

The assistant utilizes a structured `ChatPromptTemplate` comprising three distinct message layers:

```python
SYSTEM_PROMPT = """You are a helpful, professional, and concise AI Q&A Assistant.
Your responsibilities:
1. Answer the user's questions clearly, accurately, and in plain language.
2. Maintain context from earlier parts of the conversation when relevant.
3. If a question is ambiguous or lacks necessary detail, ask clarifying questions.
4. If you do not know the answer or lack verified information, admit it honestly.
5. Provide concise examples or bullet points when they make technical answers easier to understand.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])
```

### Why this design?
1. **System Message (Role & Persona):**
   - Directs the LLM to adopt a professional, concise persona.
   - Enforces grounding: explicitly instructs the model to admit when it does not know something rather than inventing facts (hallucinating).
   - Enforces formatting: encourages clear bullet points and readable examples.
2. **MessagesPlaceholder (`history`):**
   - Dynamically injects an arbitrary list of prior `BaseMessage` objects (`HumanMessage` and `AIMessage`).
   - If history is empty (Turn 1), no extra messages are injected.
   - On subsequent turns, it injects the previous dialogue so the LLM has conversational context to resolve references like *"Can you elaborate on the second point you just mentioned?"*.
3. **Human Message (`{question}`):**
   - The user's newest query is cleanly separated from system instructions, minimizing prompt injection vulnerabilities where malicious user text attempts to override system commands.

---

## 3. Explanation of the Chain Used

The assistant implements modern **LangChain Expression Language (LCEL)**:

```python
chain = prompt | model | output_parser
```

### Component Breakdown
1. **`prompt` (`ChatPromptTemplate`):**
   - Input: A dictionary `{"history": [...], "question": "..."}`.
   - Output: A formatted `PromptValue` containing the system message, conversation history, and the user's query.
2. **`model` (`ChatOpenAI`):**
   - Input: The formatted `PromptValue` from the prompt step.
   - Action: Sends a secure HTTPS POST request to OpenAI's `/v1/chat/completions` endpoint using `gpt-4o-mini` with temperature `0.7`.
   - Output: An `AIMessage` containing generation metadata, usage tokens, and the raw text completion.
3. **`output_parser` (`StrOutputParser`):**
   - Input: The `AIMessage` from the model.
   - Output: A clean, user-readable Python `str`. It extracts `AIMessage.content`, discarding internal API metadata.

### Why LCEL Instead of Legacy Chains?
- **Unified Interface:** Every component implements `invoke()`, `batch()`, `stream()`, and `astream()` out of the box.
- **Maintainability:** Declarative Unix-style pipe syntax (`|`) eliminates boilerplate callback code and nested wrapper objects.
- **Streaming & Async:** Readily supports real-time token streaming to user interfaces without rewriting chain logic.
- **Observability:** Compatible with tracing frameworks like LangSmith for debugging latency and token costs.

---

## 4. Conversation History Architecture

```
User Question ---> [Input Validation] ---> [Sliding Window Trim]
                                                   |
                                                   v
                                   [Inject History into Prompt]
                                                   |
                                                   v
                                       [LCEL Chain Execution]
                                                   |
                                                   v
[User Receives Clean Answer] <--- [Record Turn into History]
```

- **Statelessness of LLMs:** LLMs do not remember past requests. Every request is completely independent.
- **In-Memory History Buffer:** We record each `HumanMessage` and `AIMessage` pair.
- **Sliding Window Optimization:** If the history exceeds `MAX_HISTORY_TURNS` (default: 5 turns / 10 messages), the oldest messages are dropped. This prevents:
  - Context window token exhaustion.
  - Linear runaway API billing costs on long conversations.
