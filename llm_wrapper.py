import time

class AzureLLM:
    def __init__(self, client, deployment_name, temperature=0.0):
        self.client = client
        self.deployment_name = deployment_name
        self.temperature = temperature

    def invoke(self, prompt: str):
        start_time = time.time()

        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature
        )

        latency = time.time() - start_time

        content = response.choices[0].message.content

        #  Token usage
        usage = response.usage
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens

        return content, latency, prompt_tokens, completion_tokens, total_tokens