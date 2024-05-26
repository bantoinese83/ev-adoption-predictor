import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class TextGenerator:
    def __init__(self):
        self.api_key = os.getenv("API_KEY")
        self.client = OpenAI(api_key=self.api_key)
        self.conversation_history = []

    def clear_conversation_history(self):
        self.conversation_history.clear()

    def generate_text(self, user_input, max_tokens=1000):
        # Add the user's message to the conversation history
        self.conversation_history.append({"role": "user", "content": user_input})

        # Generate the AI's response
        ai_response = self.client.chat.completions.create(
            model="gpt-4",
            messages=self.conversation_history,
            max_tokens=max_tokens,
        )

        # Add the AI's response to the conversation history
        self.conversation_history.append(
            {"role": "assistant", "content": ai_response.choices[0].message.content}
        )

        # Return the actual text of the AI's response
        return ai_response.choices[0].message.content
