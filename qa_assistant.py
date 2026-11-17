#!/usr/bin/env python3
"""
LangChain Q&A Assistant
=======================
A modular, production-ready Q&A assistant built with LangChain and the OpenAI API.
Features:
- Prompt templating with system instructions and dynamic question insertion.
- Conversation history tracking using an in-memory message history buffer.
- Robust input validation and error handling for OpenAI API exceptions.
- Safe API key management via environment variables (.env).
- Interactive CLI, single-question evaluation, and automated 5-question test runner.
"""

import os
import sys
import argparse
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# LangChain core components
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI

# OpenAI API exceptions for fine-grained error handling
import openai

# ------------------------------------------------------------------------------
# System Prompt Configuration
# ------------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a helpful, professional, and concise AI Q&A Assistant.
Your responsibilities:
1. Answer the user's questions clearly, accurately, and in plain language.
2. Maintain context from earlier parts of the conversation when relevant.
3. If a question is ambiguous or lacks necessary detail, ask clarifying questions.
4. If you do not know the answer or lack verified information, admit it honestly.
5. Provide concise examples or bullet points when they make technical answers easier to understand.
"""


class QAAssistant:
    """
    Q&A Assistant managing prompt templates, LangChain LCEL chains,
    conversation history, and API error handling.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_history_turns: int = 5,
        mock_mode: bool = False,
    ):
        """
        Initialize the assistant.

        Args:
            model_name: OpenAI model identifier (e.g., 'gpt-4o-mini', 'gpt-4o').
            temperature: Creativity control between 0.0 and 1.0.
            max_history_turns: Maximum conversation turns (user+assistant pairs) to retain.
            mock_mode: If True, uses simulated responses for offline testing/demo.
        """
        # Load environment variables from .env file
        load_dotenv()

        self.mock_mode = mock_mode
        self.max_history_turns = int(os.getenv("MAX_HISTORY_TURNS", max_history_turns))
        self.model_name = model_name or os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
        self.temperature = (
            float(temperature)
            if temperature is not None
            else float(os.getenv("TEMPERATURE", 0.7))
        )

        # Retrieve API key securely from environment
        self.api_key = os.getenv("OPENAI_API_KEY")

        # Safe conversation history buffer using LangChain BaseMessage objects
        self.history_messages: List[BaseMessage] = []

        # Build prompt template
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )

        # Output parser converts ChatModel message output to clean string
        self.output_parser = StrOutputParser()

        # Initialize chain if not in mock mode and API key is present
        self.chain = None
        if not self.mock_mode and self.api_key:
            self._init_chain()

    def _init_chain(self):
        """Build the LangChain LCEL chain: prompt -> model -> output parser."""
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            api_key=self.api_key,
        )
        self.chain = self.prompt | self.llm | self.output_parser

    def _trim_history(self):
        """
        Keep only the most recent N turns in history to prevent context window overflow
        and control token consumption. Each turn is a pair: (User, Assistant).
        """
        max_messages = self.max_history_turns * 2
        if len(self.history_messages) > max_messages:
            self.history_messages = self.history_messages[-max_messages:]

    def add_user_message(self, content: str):
        """Add a human message to the conversation history."""
        self.history_messages.append(HumanMessage(content=content))

    def add_ai_message(self, content: str):
        """Add an AI response message to the conversation history."""
        self.history_messages.append(AIMessage(content=content))

    def get_history_messages(self) -> List[BaseMessage]:
        """Return the current list of conversation history messages."""
        return list(self.history_messages)

    def clear_history(self):
        """Reset conversation history."""
        self.history_messages.clear()

    @property
    def turn_count(self) -> int:
        """Return the number of completed conversation turns."""
        return len(self.history_messages) // 2

    def _generate_mock_response(self, question: str) -> str:
        """
        Generates contextual answers for offline testing or demonstration
        when live API credentials are not provided.
        """
        q_lower = question.lower()
        history_len = len(self.history_messages)

        if "what is langchain" in q_lower:
            return (
                "LangChain is an open-source framework designed to simplify the development of "
                "applications powered by Large Language Models (LLMs). It solves three primary problems "
                "for developers:\n"
                "1. Composability: Connecting prompts, models, memory, and external tools into unified pipelines.\n"
                "2. Context-Awareness: Injecting company data or databases into prompts via Retrieval-Augmented Generation (RAG).\n"
                "3. Agency: Allowing LLMs to make decisions and call tools dynamically."
            )
        elif "compare" in q_lower and ("chain" in q_lower or "agent" in q_lower):
            return (
                "Comparing Chains and Agents in LangChain:\n"
                "- Chains: Hardcoded, deterministic sequences where Step A always leads to Step B (e.g., Prompt -> Model -> Parser). "
                "Best for predictable, linear workflows.\n"
                "- Agents: Dynamic systems where the LLM acts as a reasoning engine to decide which action to take, which tool to call, "
                "and when to stop based on user input. Best for exploratory, multi-step tasks."
            )
        elif "snippet" in q_lower or "code" in q_lower:
            return (
                "Here is a minimal Python example demonstrating the LCEL pipe pattern:\n\n"
                "```python\n"
                "from langchain_core.prompts import ChatPromptTemplate\n"
                "from langchain_core.output_parsers import StrOutputParser\n"
                "from langchain_openai import ChatOpenAI\n\n"
                "prompt = ChatPromptTemplate.from_template('Explain {topic} in one sentence.')\n"
                "model = ChatOpenAI(model='gpt-4o-mini', temperature=0.7)\n"
                "parser = StrOutputParser()\n\n"
                "# The LCEL pipe composition\n"
                "chain = prompt | model | parser\n"
                "response = chain.invoke({'topic': 'LangChain Chains'})\n"
                "print(response)\n"
                "```"
            )
        elif "lcel" in q_lower:
            return (
                "LCEL (LangChain Expression Language) is a declarative, composable syntax for building chains. "
                "Key advantages include:\n"
                "1. Pipe Operator (`|`): Unix-style composition connecting components intuitively.\n"
                "2. Unified Interface: Standardized `.invoke()`, `.batch()`, and `.stream()` methods across all components.\n"
                "3. Production Readiness: Built-in support for streaming, parallel execution, fallback handling, and async out of the box."
            )
        elif "security" in q_lower or "risks" in q_lower:
            return (
                "Three critical security and reliability risks in production LLM applications:\n"
                "1. API Key Exposure & Credential Theft: Exposing keys in source code risks quota theft and unauthorized access. Always use `.env` and secret managers.\n"
                "2. Prompt Injection: Malicious user inputs overriding system instructions. Mitigate by sanitizing inputs and validating outputs.\n"
                "3. Rate Limits & Uncontrolled Costs: Sudden traffic spikes triggering API 429 errors or budget depletion. Mitigate using caching, exponential backoff, and spending caps."
            )
        else:
            return (
                f"[Demo Answer based on context with {history_len} previous messages]: "
                f"Thank you for your question about '{question}'. LangChain facilitates managing "
                "prompts, models, and conversation state to produce reliable AI responses."
            )

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Process a user question, invoke the chain, update conversation history,
        and handle potential errors.

        Args:
            question: User question string.

        Returns:
            Dictionary containing:
                - success (bool): True if answer was generated, False if error occurred.
                - answer (str): Generated answer or error explanation.
                - error_type (str, optional): Category of error if success is False.
                - turn_count (int): Total conversation turns retained in history.
        """
        # 1. Input Validation: Check for empty or whitespace-only questions
        if not question or not question.strip():
            return {
                "success": False,
                "answer": "Error: Question cannot be empty. Please provide a valid question.",
                "error_type": "ValidationError",
                "turn_count": self.turn_count,
            }

        cleaned_question = question.strip()

        # 2. Check for missing API Key (when not in mock mode)
        if not self.mock_mode and not self.api_key:
            return {
                "success": False,
                "answer": (
                    "Error: OPENAI_API_KEY is not set. Please set your API key in the .env file "
                    "or environment variable. See .env.example for instructions."
                ),
                "error_type": "ConfigurationError",
                "turn_count": self.turn_count,
            }

        # 3. Offline / Demo Mode Execution
        if self.mock_mode:
            answer = self._generate_mock_response(cleaned_question)
            # Record user question and AI answer in history
            self.add_user_message(cleaned_question)
            self.add_ai_message(answer)
            self._trim_history()
            return {
                "success": True,
                "answer": answer,
                "error_type": None,
                "turn_count": self.turn_count,
            }

        # 4. Live LangChain Execution with Comprehensive Error Handling
        try:
            if self.chain is None:
                self._init_chain()

            # Retrieve current conversation history
            history_messages = self.get_history_messages()

            # Invoke chain using LCEL
            response = self.chain.invoke(
                {
                    "history": history_messages,
                    "question": cleaned_question,
                }
            )

            # Update conversation history with the successful exchange
            self.add_user_message(cleaned_question)
            self.add_ai_message(response)
            self._trim_history()

            return {
                "success": True,
                "answer": response,
                "error_type": None,
                "turn_count": self.turn_count,
            }

        except openai.AuthenticationError as e:
            return {
                "success": False,
                "answer": (
                    "Authentication Error: Invalid or expired OpenAI API key. "
                    "Please verify the OPENAI_API_KEY in your .env file."
                ),
                "error_type": "AuthenticationError",
                "details": str(e),
                "turn_count": self.turn_count,
            }

        except openai.RateLimitError as e:
            return {
                "success": False,
                "answer": (
                    "Rate Limit Error: You have exceeded your OpenAI rate limit or credit quota. "
                    "Please check your plan and billing details at platform.openai.com."
                ),
                "error_type": "RateLimitError",
                "details": str(e),
                "turn_count": self.turn_count,
            }

        except openai.APIConnectionError as e:
            return {
                "success": False,
                "answer": (
                    "Network Error: Unable to reach the OpenAI API servers. "
                    "Please check your internet connection and proxy/firewall settings."
                ),
                "error_type": "APIConnectionError",
                "details": str(e),
                "turn_count": self.turn_count,
            }

        except openai.APIError as e:
            return {
                "success": False,
                "answer": f"OpenAI API Error: {str(e)}",
                "error_type": "APIError",
                "details": str(e),
                "turn_count": self.turn_count,
            }

        except Exception as e:
            return {
                "success": False,
                "answer": f"Unexpected Error: An internal error occurred ({type(e).__name__}: {str(e)})",
                "error_type": "GeneralException",
                "details": str(e),
                "turn_count": self.turn_count,
            }


# ------------------------------------------------------------------------------
# Test Suite Runner
# ------------------------------------------------------------------------------
def run_test_suite(assistant: QAAssistant):
    """
    Executes a structured test suite covering:
    - 5 progressive test questions demonstrating conversation history.
    - Edge case validation: empty input.
    - Error handling demonstration: missing/invalid API key.
    """
    print("\n" + "=" * 70)
    print("  LangChain Q&A Assistant - Comprehensive Test Suite")
    print("=" * 70)
    print(f"Model: {assistant.model_name}")
    print(f"Temperature: {assistant.temperature}")
    print(f"Max History Turns: {assistant.max_history_turns}")
    print(f"Execution Mode: {'MOCK/DEMO' if assistant.mock_mode else 'LIVE OPENAI API'}")
    print("=" * 70 + "\n")

    test_questions = [
        (
            1,
            "What is LangChain and what problem does it solve for software developers?",
            "Tests core domain understanding and prompt template rendering.",
        ),
        (
            2,
            "Can you compare the concepts of 'Chains' and 'Agents' that you just mentioned?",
            "Tests multi-turn conversation memory by referring back to terms in Question 1.",
        ),
        (
            3,
            "How does LCEL (LangChain Expression Language) make building these chains easier?",
            "Tests specialized technical explanation and engineering benefits.",
        ),
        (
            4,
            "Can you show a minimal Python code snippet illustrating that LCEL pipe pattern?",
            "Tests code generation grounded in previous discussion turns.",
        ),
        (
            5,
            "What are three critical security and reliability risks when deploying this in production?",
            "Tests understanding of practical engineering risks, error handling, and API safety.",
        ),
    ]

    for q_num, question, rationale in test_questions:
        print(f"[TEST {q_num}]")
        print(f"Question:  {question}")
        print(f"Rationale: {rationale}")
        print("-" * 70)

        result = assistant.ask(question)

        if result["success"]:
            print(f"Status:    SUCCESS")
            print(f"History:   {result['turn_count']} turn(s) active in memory")
            print(f"\nAnswer:\n{result['answer']}\n")
        else:
            print(f"Status:    FAILED ({result.get('error_type')})")
            print(f"Message:   {result['answer']}\n")

        print("=" * 70 + "\n")

    # Edge Case Test 1: Empty Question
    print("[EDGE CASE TEST 1] Empty/Whitespace-only Question Validation")
    empty_q = "    "
    print(f"Question:  '{empty_q}'")
    result_empty = assistant.ask(empty_q)
    print(f"Success:   {result_empty['success']} (Expected: False)")
    print(f"Error:     {result_empty['answer']}\n")
    print("=" * 70 + "\n")

    # Edge Case Test 2: Error Handling with invalid key (live mode test)
    print("[EDGE CASE TEST 2] Authentication Error Handling Simulation")
    broken_assistant = QAAssistant(mock_mode=False)
    broken_assistant.api_key = "sk-invalid-test-key-that-will-fail-gracefully"
    broken_assistant._init_chain()
    result_auth = broken_assistant.ask("Hello, are you there?")
    print(f"Input:     'Hello, are you there?' with invalid API key")
    print(f"Success:   {result_auth['success']} (Expected: False)")
    print(f"Error:     {result_auth['answer']}")
    print(f"Category:  {result_auth.get('error_type')}\n")
    print("=" * 70 + "\n")

    print("Test Suite Execution Completed.")


# ------------------------------------------------------------------------------
# Interactive CLI
# ------------------------------------------------------------------------------
def run_interactive_mode(assistant: QAAssistant):
    """Start an interactive command-line Q&A chat session."""
    print("\n" + "=" * 60)
    print("  LangChain Q&A Assistant - Interactive Chat")
    print("=" * 60)
    print("Type your question and press Enter.")
    print("Commands:")
    print("  'exit' or 'quit' - End session")
    print("  'clear'          - Clear conversation history")
    print("  'history'        - View active conversation history")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "q"):
                print("\nGoodbye!")
                break

            if user_input.lower() == "clear":
                assistant.clear_history()
                print("\n[Conversation history cleared]\n")
                continue

            if user_input.lower() == "history":
                msgs = assistant.get_history_messages()
                print(f"\n--- History ({len(msgs)} messages / {len(msgs)//2} turns) ---")
                for i, msg in enumerate(msgs, 1):
                    sender = "User" if isinstance(msg, HumanMessage) else "Assistant"
                    snippet = msg.content[:80] + ("..." if len(msg.content) > 80 else "")
                    print(f"  {i}. [{sender}]: {snippet}")
                print("--------------------------------------------------\n")
                continue

            result = assistant.ask(user_input)

            if result["success"]:
                print(f"\nAssistant:\n{result['answer']}\n")
            else:
                print(f"\n[Error - {result.get('error_type')}]: {result['answer']}\n")

        except KeyboardInterrupt:
            print("\n\nSession terminated by user. Goodbye!")
            break


# ------------------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="LangChain Q&A Assistant with conversation history and error handling."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run the automated 5-question test suite and edge case evaluations.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo/mock mode without requiring an active OpenAI API key.",
    )
    parser.add_argument(
        "--question",
        "-q",
        type=str,
        help="Ask a single question and output the answer.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specify OpenAI model (default: gpt-4o-mini or env OPENAI_MODEL_NAME).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Specify model temperature (default: 0.7 or env TEMPERATURE).",
    )

    args = parser.parse_args()

    # Determine mock mode:
    # Use mock if --demo is specified, or if running --test and OPENAI_API_KEY is not set
    load_dotenv()
    has_api_key = bool(os.getenv("OPENAI_API_KEY"))
    mock_mode = args.demo or (args.test and not has_api_key)

    assistant = QAAssistant(
        model_name=args.model,
        temperature=args.temperature,
        mock_mode=mock_mode,
    )

    if args.test:
        run_test_suite(assistant)
    elif args.question:
        result = assistant.ask(args.question)
        if result["success"]:
            print(result["answer"])
        else:
            print(f"Error ({result.get('error_type')}): {result['answer']}", file=sys.stderr)
            sys.exit(1)
    else:
        run_interactive_mode(assistant)


if __name__ == "__main__":
    main()
