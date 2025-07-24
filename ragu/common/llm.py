import importlib.util
import logging

from openai import OpenAI
from ragu.common.decorator import no_throw
from langchain_openai import ChatOpenAI


class BaseLLM:
    def __init__(self):
        self.statistics = {
            "total_tokens": 0,
            "completion_tokens": 0,
            "prompt_tokens": 0
        }

    @no_throw
    def generate(self, *args, **kwargs):
        ...

    def get_statistics(self):
        return self.statistics

    def reset_statistics(self):
        self.statistics = {
            "total_tokens": 0,
            "completion_tokens": 0,
            "prompt_tokens": 0,
        }

class LocalLLM(BaseLLM):
    def __init__(self, model_name: str, *args, **kwargs):
        spec = importlib.util.find_spec("transformers")
        if spec is not None:
            transformers = importlib.import_module("transformers")
        else:
            raise ImportError("transformers is not installed. Please install it using pip install transformers or compile from source_entity.")

        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **kwargs,
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )
        super().__init__()

    def generate(self, query: str, system_prompt: str, *args, **kwargs):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        result = self.pipe(messages, **kwargs)
        return result[0]['generated_text'][2]['content'] if isinstance(result, list) else result


class RemoteLLM(BaseLLM):
    """
    A class representing a remote Large Language Model (LLM) interface.

    Attributes:
        model_name (str): Name of the model to be used.
        trials (int): Number of retry attempts in case of failure.
        client (OpenAI): OpenAI client instance for interacting with the model.
    """

    def __init__(
            self, model_name: str, base_url: str, api_token: str, trials: int = 2, **kwargs
    ):
        """
        Initializes the RemoteLLM instance.

        :param model_name: Name of the LLM model.
        :param base_url: Base URL for the API endpoint.
        :param api_token: API token for authentication.
        :param trials: Number of retry attempts in case of failure (default is 2).
        :param kwargs: Additional keyword arguments for OpenAI client configuration.
        """
        super().__init__()

        assert trials > 0, "Number of trials must be greater than 0"

        self.trials = trials
        self.model_name = model_name
        self.client = OpenAI(base_url=base_url, api_key=api_token, **kwargs)

    @no_throw
    def generate(
            self,
            queries: str | list[str],
            system_prompt: str,
            model_name: str = None,
            **kwargs
    ):
        """
        Generates a response from the LLM based on user queries.

        :param queries: A single query string or a list of query strings.
        :param system_prompt: System-level prompt to provide context.
        :param model_name: Optional model name override.
        :param kwargs: Additional keyword arguments for the API request.
        :return: A response string if input is a string, otherwise a list of response strings.
        """
        if isinstance(queries, str):
            queries = [queries]

        return [self._get_response(query, system_prompt, **kwargs) for query in queries]

    def _get_response(self, query: str, system_prompt: str, **kwargs):
        """
        Sends a request to the OpenAI API and retrieves a response.

        :param query: User's query input.
        :param system_prompt: System prompt for context.
        :param kwargs: Additional keyword arguments for the API request.
        :return: The model's response as a string, or None if no valid response is obtained.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        for _ in range(self.trials):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    **kwargs
                )
                print("OOOOOOOOOOOOO", response)
                print("PPPPPPPPPPPPP")

                if not response.choices:
                    continue

                usage = response.usage
                if usage:
                    self.statistics["total_tokens"] += usage.total_tokens or 0
                    self.statistics["completion_tokens"] += usage.completion_tokens or 0
                    self.statistics["prompt_tokens"] += usage.prompt_tokens or 0

                return response.choices[0].message.content

            except Exception as e:
                print("EEEERRRROOOORRRR: ", e)
                logging.error(f"Failed to generate response: {e}")
                return None

        print("FFFFFF Just None")
        return None


class VLLMClient(BaseLLM):
    """
    A client for interacting with vLLM-based language models.

    Attributes:
        engine (LLM): The vLLM engine instance.
        sampling_params (SamplingParams): Parameters for controlling text generation.
    """

    def __init__(
        self,
        model_name: str,
        sampling_params: dict = None,
        **vllm_kwargs,
    ):
        """
        Initializes the vLLM client.

        :param model_name: Name of the LLM model.
        :param sampling_params: Dictionary of sampling parameters (default: max_tokens=2048).
        """
        super().__init__()

        spec = importlib.util.find_spec("vllm")
        if spec is not None:
            vllm = importlib.import_module("vllm")
        else:
            raise ImportError("vLLM is not installed. Please install it using pip install vllm or compile from source_entity.")

        self.engine = vllm.LLM(
            model=model_name,
            **vllm_kwargs,
        )

        self.sampling_params = vllm.SamplingParams(**(sampling_params or {"max_tokens": 2048}))

    @no_throw
    def generate(self, queries: str | list[str], system_prompt: str, remove_none: bool = True, **kwargs):
        """
        Generates responses from the vLLM model based on user queries.

        :param remove_none:
        :param queries: A single query string or a list of query strings.
        :param system_prompt: System-level prompt to provide context.
        :param kwargs: Additional keyword arguments for the API request.
        :return: A single response string if input is a string, otherwise a list of response strings.
        """
        queries = [queries] if isinstance(queries, str) else queries

        batched_prompts = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ]
            for query in queries
        ]

        outputs = self.engine.chat(batched_prompts, self.sampling_params)
        responses = [output.outputs[0].text for output in outputs]

        try:
            self.statistics["completion_tokens"] += sum([len(output.outputs[0].token_ids) for output in outputs])
        except Exception as e:
            ...

        return responses
