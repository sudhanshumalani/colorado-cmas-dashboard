"""
Claude Interface Module
Main Claude API wrapper with prompt caching and conversation management.
"""

import anthropic
from typing import List, Dict, Optional, Generator
import json


class ClaudeInterface:
    """Manage Claude API interactions with optimization"""

    def __init__(self, api_key: str):
        """Initialize Claude client"""
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-3-5-sonnet-20241022"  # Latest model with caching

    def chat(
        self,
        user_message: str,
        context_data: str,
        schema: str,
        conversation_history: List[Dict] = None,
        stream: bool = False
    ) -> str:
        """
        Send message to Claude with context

        Args:
            user_message: User's question
            context_data: Relevant data from database/vector store
            schema: Database schema (will be cached)
            conversation_history: Previous messages
            stream: Whether to stream response

        Returns:
            Claude's response text
        """

        # Build system prompt with caching
        system_prompt = self._build_system_prompt(schema, context_data)

        # Build messages
        messages = []

        # Add conversation history (last 10 exchanges)
        if conversation_history:
            messages.extend(conversation_history[-20:])  # Last 10 exchanges = 20 messages

        # Add current user message
        messages.append({
            "role": "user",
            "content": user_message
        })

        try:
            if stream:
                # Streaming response
                return self._stream_response(system_prompt, messages)
            else:
                # Regular response
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    system=system_prompt,
                    messages=messages
                )
                return message.content[0].text

        except Exception as e:
            error_msg = str(e)
            if "overloaded" in error_msg.lower():
                return "⚠️ Claude API is currently overloaded. Please try again in a moment."
            elif "rate_limit" in error_msg.lower():
                return "⚠️ Rate limit exceeded. Please wait a moment before asking another question."
            else:
                return f"⚠️ Error: {error_msg[:200]}"

    def _build_system_prompt(self, schema: str, context_data: str) -> List[Dict]:
        """
        Build system prompt with prompt caching

        Cached blocks:
        - Schema (static, changes rarely)
        - Guidelines (static)

        Not cached:
        - Context data (changes per query)
        """

        system_blocks = [
            # Block 1: Schema (CACHED - saves tokens on repeated calls)
            {
                "type": "text",
                "text": f"""You are an expert educational data analyst helping users understand Colorado charter school CMAS performance data.

DATABASE SCHEMA:
{schema}

Your role is to:
1. Answer questions accurately using the provided data
2. Provide insights and analysis when asked
3. Cite specific schools and numbers
4. Be honest when data is insufficient
5. Suggest follow-up questions when relevant""",
                "cache_control": {"type": "ephemeral"}  # Cache this block
            },

            # Block 2: Guidelines (CACHED)
            {
                "type": "text",
                "text": """ANALYSIS GUIDELINES:

Performance Terciles:
- Top Third = Schools performing ABOVE the trendline for their FRL%
- Middle Third = Schools performing NEAR the trendline
- Bottom Third = Schools performing BELOW the trendline
- This is about performance relative to demographics, NOT absolute scores

Important Terminology:
- "Single Site Charter School" = Independent charter, not part of a network/CMO
- FRL (Free/Reduced Lunch) = Poverty indicator (0-100%)
- Network = Charter Management Organization (CMO) like KIPP, DSST, etc.

Response Format:
- Be concise (2-4 sentences for simple questions, more for analysis)
- Always cite specific school names when possible
- Provide exact numbers, not approximations
- If asked about a specific school, search by name carefully
- For "why" questions, provide data-driven insights

Quality Checks:
- Verify school names match the data exactly
- Double-check numbers before citing them
- If uncertain, say so rather than guessing
- Don't make assumptions beyond what the data shows""",
                "cache_control": {"type": "ephemeral"}  # Cache this block
            },

            # Block 3: Current context data (NOT CACHED - changes per query)
            {
                "type": "text",
                "text": f"""RELEVANT DATA FOR THIS QUERY:

{context_data}

Answer the user's question based on this data. Be specific and cite schools by name."""
            }
        ]

        return system_blocks

    def _stream_response(self, system_prompt: List[Dict], messages: List[Dict]) -> Generator:
        """Stream response from Claude"""

        try:
            with self.client.messages.stream(
                model=self.model,
                max_tokens=2000,
                system=system_prompt,
                messages=messages
            ) as stream:
                for chunk in stream.text_stream:
                    yield chunk

        except Exception as e:
            yield f"\n\n⚠️ Streaming error: {str(e)[:200]}"

    def analyze_with_context(
        self,
        question: str,
        sql_results: List[Dict],
        schema: str,
        conversation_history: List[Dict] = None
    ) -> str:
        """
        Analyze SQL results to answer a question

        Args:
            question: User's original question
            sql_results: Results from SQL query
            schema: Database schema
            conversation_history: Previous messages

        Returns:
            Natural language analysis
        """

        # Format SQL results as context
        if not sql_results:
            context_data = "No results found from the database query."
        else:
            context_data = "QUERY RESULTS:\n\n"
            context_data += json.dumps(sql_results, indent=2)

        # Get analysis from Claude
        analysis_prompt = f"""Based on the query results above, answer this question:

{question}

Provide a clear, concise answer with specific school names and numbers."""

        return self.chat(
            user_message=analysis_prompt,
            context_data=context_data,
            schema=schema,
            conversation_history=conversation_history,
            stream=False
        )

    def generate_follow_up_questions(
        self,
        question: str,
        answer: str,
        context: str
    ) -> List[str]:
        """
        Generate relevant follow-up questions

        Args:
            question: Original question
            answer: Response given
            context: Data context

        Returns:
            List of 3 follow-up questions
        """

        try:
            prompt = f"""Based on this conversation:

Question: {question}
Answer: {answer}

Context: {context[:500]}

Generate 3 relevant follow-up questions the user might want to ask. Return ONLY the questions, one per line, no numbering."""

            message = self.client.messages.create(
                model=self.model,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )

            response = message.content[0].text.strip()
            questions = [q.strip() for q in response.split('\n') if q.strip()]

            return questions[:3]

        except:
            # Fallback generic questions
            return [
                "Can you show me more details about these schools?",
                "How do these schools compare to the state average?",
                "What factors might explain this performance?"
            ]


if __name__ == "__main__":
    # Test the Claude interface
    import os
    from dotenv import load_dotenv
    from data_processor import DataProcessor

    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        print("❌ No API key found")
        exit(1)

    # Initialize
    claude = ClaudeInterface(api_key)
    processor = DataProcessor()
    schema = processor.get_schema()

    # Test chat
    print("🤖 Testing Claude chat:")
    context = """
QUERY RESULTS:

School: Vega Collegiate Academy
District: Adams-Arapahoe 28J
FRL: 98%
ELA Performance: 18%
Math Performance: 16%
"""

    response = claude.chat(
        user_message="Which district is Vega Collegiate Academy from?",
        context_data=context,
        schema=schema
    )

    print(f"\nResponse: {response}")

    # Test follow-up questions
    print("\n💡 Suggested follow-up questions:")
    follow_ups = claude.generate_follow_up_questions(
        question="Which district is Vega Collegiate Academy from?",
        answer=response,
        context=context
    )

    for i, q in enumerate(follow_ups, 1):
        print(f"{i}. {q}")
