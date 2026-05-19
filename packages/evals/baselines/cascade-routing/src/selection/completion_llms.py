from langchain_anthropic import AnthropicLLM
from langchain_together import Together
from langchain_openai import OpenAI
from aiohttp import ClientSession
from langchain_core.runnables import get_config_list
from langchain_huggingface import HuggingFacePipeline

from typing import (
    List,
    cast,
    Dict,
    Any,
    Optional,
)
from langchain_core.outputs import Generation, LLMResult
import inspect

# This code contains classes that were almost completely copied from the respective libraries
# and modified to be compatible with the API framework used in this project.


class HuggingFacePipelineFixed(HuggingFacePipeline):
    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager= None,
        **kwargs: Any,
    ) -> LLMResult:
        # List to hold all results
        text_generations: List[str] = []
        pipeline_kwargs = kwargs.get("pipeline_kwargs", {})
        skip_prompt = kwargs.get("skip_prompt", False)

        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i : i + self.batch_size]

            # Process batch of prompts
            responses = self.pipeline(
                batch_prompts,
                **pipeline_kwargs,
            )

            # Process each response in the batch
            for j, response in enumerate(responses):
                if isinstance(response, list):
                    # if model returns multiple generations, pick the top one
                    response = response[0]
                if skip_prompt:
                    text = text[len(batch_prompts[j]) :]
                # Append the processed text to results
                text_generations.append(response)

        return LLMResult(
            generations=[[Generation(text=response['content'], generation_info=response)] for response in text_generations]
        )
    async def abatch(
        self,
        inputs,
        config=None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> List[str]:
        if not inputs:
            return []
        config = get_config_list(config, len(inputs))
        max_concurrency = config[0].get("max_concurrency")

        if max_concurrency is None:
            try:
                llm_result = await self.agenerate_prompt(
                    [self._convert_input(input) for input in inputs],
                    callbacks=[c.get("callbacks") for c in config],
                    tags=[c.get("tags") for c in config],
                    metadata=[c.get("metadata") for c in config],
                    run_name=[c.get("run_name") for c in config],
                    **kwargs,
                )
                return [g[0] for g in llm_result.generations]
            except Exception as e:
                if return_exceptions:
                    return cast(List[str], [e for _ in inputs])
                else:
                    raise e
        else:
            batches = [
                inputs[i : i + max_concurrency]
                for i in range(0, len(inputs), max_concurrency)
            ]
            config = [{**c, "max_concurrency": None} for c in config]  # type: ignore[misc]
            return [
                output
                for i, batch in enumerate(batches)
                for output in await self.abatch(
                    batch,
                    config=config[i * max_concurrency : (i + 1) * max_concurrency],
                    return_exceptions=return_exceptions,
                    **kwargs,
                )
            ]

class AnthropicLLMCompletion(AnthropicLLM):
    async def _acall(self, prompt, stop=None, run_manager=None, **kwargs) -> str:
        """Call out to Anthropic's completion endpoint asynchronously."""
        if self.streaming:
            completion = ""
            async for chunk in self._astream(
                prompt=prompt, stop=stop, run_manager=run_manager, **kwargs
            ):
                completion += chunk.text
            return completion

        stop = self._get_anthropic_stop(stop)
        params = {**self._default_params, **kwargs}

        response = await self.async_client.completions.create(
            prompt=self._wrap_prompt(prompt),
            stop_sequences=stop,
            **params,
        )
        return {
            'content': response.completion,
            'usage_metadata': {
                'input_tokens': self.count_tokens(prompt),
                'output_tokens': self.count_tokens(response.completion),
            }
        }


class TogetherLLMCompletion(Together):
    echo : bool = False
    stop : Optional[List[str]] = None

    @property
    def default_params(self) -> Dict[str, Any]:
        """Return the default parameters for the Together model.

        Returns:
            A dictionary containing the default parameters.
        """
        kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "repetition_penalty": self.repetition_penalty,
            "logprobs": self.logprobs,
            "echo": self.echo
        }
        return kwargs
    
    async def _acall(self, prompt, stop=None, run_manager=None, **kwargs) -> str:
        """Call Together model to get predictions based on the prompt.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.
            run_manager: The CallbackManager for LLM run, it's not used at the moment.
            **kwargs: Additional parameters to pass to the model.

        Returns:
            The string generated by the model.
        """
        headers = {
            "Authorization": f"Bearer {self.together_api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        stop_to_use = stop[0] if stop and len(stop) == 1 else stop
        if self.stop is not None:
            stop_to_use = self.stop

        payload = {
            **self.default_params,
            "prompt": prompt,
            "stop": stop_to_use,
            **kwargs,
        }
        # filter None values to not pass them to the http payload
        payload = {k: v for k, v in payload.items() if v is not None}
        async with ClientSession() as session:
            async with session.post(
                self.base_url, json=payload, headers=headers
            ) as response:
                if response.status >= 500:
                    raise Exception(f"Together Server: Error {response.status}")
                elif response.status >= 400:
                    raise ValueError(
                        f"Together received an invalid payload: {response.text}"
                    )
                elif response.status != 200:
                    raise Exception(
                        f"Together returned an unexpected response with status "
                        f"{response.status}: {response.text}"
                    )
                response_json = await response.json()

                output = self._format_output(response_json)
                input_tokens = response_json['usage']['prompt_tokens']
                output_tokens = response_json['usage']['completion_tokens']
                logprobs = None
                if len(response_json['prompt']) > 0:
                    logprobs = response_json['prompt'][0].get('logprobs')
                return {
                    'content': output,
                    'usage_metadata': {
                        'input_tokens': input_tokens,
                        'output_tokens': output_tokens
                    },
                    'response_metadata': {
                        'logprobs': logprobs
                    }
                }

    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs: Any,
    ) -> LLMResult:
        """Async run the LLM on the given prompt and input."""
        generations = []
        new_arg_supported = inspect.signature(self._acall).parameters.get("run_manager")
        for prompt in prompts:
            text = (
                await self._acall(prompt, stop=stop, run_manager=run_manager, **kwargs)
                if new_arg_supported
                else await self._acall(prompt, stop=stop, **kwargs)
            )
            generations.append([Generation(text=text['content'], generation_info=text)])
        return LLMResult(generations=generations)
    
    async def abatch(
        self,
        inputs,
        config=None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> List[str]:
        if not inputs:
            return []
        config = get_config_list(config, len(inputs))
        max_concurrency = config[0].get("max_concurrency")

        if max_concurrency is None:
            try:
                llm_result = await self.agenerate_prompt(
                    [self._convert_input(input) for input in inputs],
                    callbacks=[c.get("callbacks") for c in config],
                    tags=[c.get("tags") for c in config],
                    metadata=[c.get("metadata") for c in config],
                    run_name=[c.get("run_name") for c in config],
                    **kwargs,
                )
                return [g[0] for g in llm_result.generations]
            except Exception as e:
                if return_exceptions:
                    return cast(List[str], [e for _ in inputs])
                else:
                    raise e
        else:
            batches = [
                inputs[i : i + max_concurrency]
                for i in range(0, len(inputs), max_concurrency)
            ]
            config = [{**c, "max_concurrency": None} for c in config]  # type: ignore[misc]
            return [
                output
                for i, batch in enumerate(batches)
                for output in await self.abatch(
                    batch,
                    config=config[i * max_concurrency : (i + 1) * max_concurrency],
                    return_exceptions=return_exceptions,
                    **kwargs,
                )
            ]

class OpenAICompletion(OpenAI):
    async def abatch(self, inputs, config=None, *, return_exceptions=False, **kwargs):
        if not inputs:
            return []
        config = get_config_list(config, len(inputs))
        max_concurrency = config[0].get("max_concurrency")

        if max_concurrency is None:
            try:
                llm_result = await self.agenerate_prompt(
                    [self._convert_input(input) for input in inputs],
                    callbacks=[c.get("callbacks") for c in config],
                    tags=[c.get("tags") for c in config],
                    metadata=[c.get("metadata") for c in config],
                    run_name=[c.get("run_name") for c in config],
                    **kwargs,
                )
                outputs = [g[0].text for g in llm_result.generations]
                token_usage = [
                    {
                        "input_tokens": llm_result.llm_output['token_usage']['prompt_tokens'] / len(outputs),
                        "output_tokens": llm_result.llm_output['token_usage']['completion_tokens'] / len(outputs),
                    } for _ in outputs
                ]
                return [
                    {
                        "content": output,
                        "usage_metadata": token_usage[i],
                    }
                    for i, output in enumerate(outputs)
                ]
            except Exception as e:
                if return_exceptions:
                    return cast(List[str], [e for _ in inputs])
                else:
                    raise e
        else:
            batches = [
                inputs[i : i + max_concurrency]
                for i in range(0, len(inputs), max_concurrency)
            ]
            config = [{**c, "max_concurrency": None} for c in config]  # type: ignore[misc]
            return [
                output
                for i, batch in enumerate(batches)
                for output in await self.abatch(
                    batch,
                    config=config[i * max_concurrency : (i + 1) * max_concurrency],
                    return_exceptions=return_exceptions,
                    **kwargs,
                )
            ]