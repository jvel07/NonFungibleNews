from langchain_ollama import ChatOllama

class ChatLLM:
    def __init__(self, model="llama3.1:8b-instruct-fp16", temperature=0.8, num_predict=256, num_ctx=1024*8):
        self.llm = ChatOllama(
            model=model,
            temperature=temperature,
            num_predict=num_predict,
            num_ctx=num_ctx
        )
        self.system_prompt = "You are a helpful person."

    def set_system_prompt(self, prompt):
        self.system_prompt = prompt
        return self

    def ask(self, question):
        if isinstance(question, str):
            return self.llm.invoke(question)

        elif isinstance(question, list):
            messages = [("system", self.system_prompt)] + question
            return self.llm.invoke(messages)

    def chat(self, messages):
        return self.ask(messages)


# use
# chat = ChatLLM(model='deepseek-r1:14b')

# answer = chat.ask("Who wrote el principe?")

# custom system prompt
# chat.set_system_prompt("You are an expert on books")
# answer = chat.ask("Who wrote 100 anos de soledad?")

# chat history
# messages = [
#     ("human", "What color is the sky?"),
#     ("assistant", "The sky is blue."),
#     ("human", "Why?")
# ]
# answer = chat.chat(messages)
# print(answer)