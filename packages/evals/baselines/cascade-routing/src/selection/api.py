import os
from tqdm import tqdm

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_together import ChatTogether
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_huggingface import HuggingFacePipeline
import torch
import time
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextGenerationPipeline

from .completion_llms import *

import warnings
warnings.filterwarnings("ignore", message=".*this flag is only used in sample-based generation modes.*")

class LogProbsTextGenerationPipeline(TextGenerationPipeline):
    def __call__(self, input_texts, *args, **kwargs):
        """
        Process a batch of input texts and return generated text, log probabilities,
        and tokenized representations for both input and output tokens.

        Args:
            input_texts (list): List of input strings to process.

        Returns:
            list[dict]: A list of dictionaries, each containing:
                - `generated_text`: The generated text string.
                - `all_tokens`: Combined list of input and output tokens.
                - `log_probs`: Log probabilities for each token in `all_tokens`.
        """
        # Tokenize all inputs for batching
        inputs = self.tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)

        # Forward pass to compute logits for input tokens
        with torch.no_grad():
            input_outputs = self.model(**inputs)
            input_logits = input_outputs.logits

        # Compute log probabilities for input tokens
        input_log_probs = F.log_softmax(input_logits, dim=-1)

        # Generate output tokens in a batched manner
        model_outputs = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            output_scores=True,
            return_dict_in_generate=True,
            **self._forward_params
        )

        # Extract sequences and scores (logits)
        generated_ids = model_outputs.sequences
        logits = model_outputs.scores

        # Process each input in the batch
        results = []
        for batch_idx in range(len(input_texts)):
            input_length = inputs.input_ids[batch_idx].shape[0]

            # Input tokens and their log probabilities
            input_tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[batch_idx].tolist())
            input_log_probs_batch = [0] + [
                input_log_probs[batch_idx, i, token_id].item()
                for i, token_id in enumerate(inputs.input_ids[batch_idx][1:])
            ]

            # Generated tokens and their log probabilities
            generated_tokens = self.tokenizer.convert_ids_to_tokens(generated_ids[batch_idx].tolist())
            generated_log_probs = []
            for i, step_logits in enumerate(logits):
                step_log_probs = F.log_softmax(step_logits, dim=-1)
                token_id = generated_ids[batch_idx, input_length + i]
                token_log_prob = step_log_probs[batch_idx, token_id].item()
                generated_log_probs.append(token_log_prob)

            # Combine input and output tokens and log probabilities
            all_tokens = input_tokens + generated_tokens[input_length:]
            log_probs = input_log_probs_batch + generated_log_probs

            # Decode the generated text
            generated_text = self.tokenizer.decode(generated_ids[batch_idx], skip_special_tokens=True)
            results.append({
                "content": generated_text,
                "tokens": all_tokens,
                "logprobs": log_probs,
                "usage_metadata": {
                    "input_tokens": len(input_tokens),
                    "output_tokens": len(generated_tokens) - len(input_tokens)
                }
            })
        return results



class APIQuery:
    def __init__(self, model, 
                 timeout=30, 
                 temperature=0, 
                 max_tokens=256,
                 return_logprobs=False, 
                 api='openai', 
                 chat=True, 
                 max_retries=20,
                 requests_per_second=30, 
                 check_every_n_seconds=0.1,
                 read_cost=None,  
                 write_cost=None, 
                 sequential=False,
                 **kwargs):
        """
        Initializes an instance of the API class.
        Args:
            model (str): The model to query.
            timeout (int, optional): The timeout value in seconds. Defaults to 30.
            temperature (int, optional): The temperature value. Defaults to 0.
            max_tokens (int, optional): The maximum number of tokens. Defaults to 256.
            return_logprobs (bool, optional): Whether to return log probabilities. Defaults to False.
            api (str, optional): The API to be used, one of "openai", "together", "huggingface", "gemini", "claude". Defaults to 'openai'.
            chat (bool, optional): Whether to enable chat mode. Defaults to True.
            max_retries (int, optional): The maximum number of retries. Defaults to 20.
            requests_per_second (int, optional): The number of requests per second. Defaults to 30.
            check_every_n_seconds (float, optional): The interval for checking rate limits. Defaults to 0.1.
            read_cost (float, optional): The cost of read operations. Defaults to None.
            write_cost (float, optional): The cost of write operations. Defaults to None.
            sequential (bool, optional): Whether to run queries sequentially. Defaults to False.
            **kwargs: Additional keyword arguments for the API model.
        Returns:
            None
        """
        self.model = model
        self.return_logprobs = return_logprobs
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs
        self.tokenizer = None
        if read_cost is None:
            self.read_cost = 1
            self.write_cost = 1
        else:
            self.read_cost = read_cost
            self.write_cost = write_cost
        self.max_retries = max_retries
        self.sequential = sequential
        self.api = api
        self.chat = chat
        self.timeout = timeout
        self.rate_limiter = InMemoryRateLimiter(requests_per_second=requests_per_second, 
                                                check_every_n_seconds=check_every_n_seconds, 
                                                max_bucket_size=requests_per_second)
        self.initialize_api_model()

    def initialize_api_model(self):
        """
        Initializes the API model based on the specified API and chat settings.
        Raises:
            ValueError: If the specified API or chat settings are not supported.
        Returns:
            None
        """
        if self.api == 'openai' and self.chat:
            self.api_model = ChatOpenAI(model=self.model, max_retries=self.max_retries, 
                                        temperature=self.temperature,
                                        timeout=self.timeout, max_tokens=self.max_tokens, 
                                        rate_limiter=self.rate_limiter, 
                                        api_key=os.getenv('OPENAI_API_KEY'), seed=0, 
                                        stream_usage=True, **self.kwargs)
        if self.api == 'hyperbolic' and self.chat:
            self.api_model = ChatOpenAI(model=self.model, max_retries=self.max_retries, 
                                        temperature=self.temperature,
                                        timeout=self.timeout, max_tokens=self.max_tokens, 
                                        rate_limiter=self.rate_limiter, 
                                        api_key=os.getenv('HYPERBOLIC_API_KEY'), seed=0,
                                        base_url="https://api.hyperbolic.xyz/v1", 
                                        stream_usage=True, **self.kwargs)
        elif self.api == 'openai' and not self.chat:
            self.api_model = OpenAICompletion(model=self.model, temperature=self.temperature,
                                              max_tokens=self.max_tokens, 
                                              api_key=os.getenv('OPENAI_API_KEY'), seed=0, 
                                              **self.kwargs)
        elif self.api == 'hyperbolic' and not self.chat:
            self.api_model = OpenAICompletion(model=self.model, temperature=self.temperature,
                                              max_tokens=self.max_tokens, 
                                              api_key=os.getenv('HYPERBOLIC_API_KEY'), seed=0,
                                              base_url="https://api.hyperbolic.xyz/v1", 
                                              **self.kwargs)
        elif self.api == 'anthropic' and self.chat:
            self.api_model = ChatAnthropic(model_name=self.model, temperature=self.temperature,
                                           timeout=self.timeout, max_tokens=self.max_tokens, 
                                           api_key=os.getenv('ANTHROPIC_API_KEY'), 
                                           stream_usage=True, **self.kwargs)
        elif self.api == 'anthropic' and not self.chat:
            self.api_model = AnthropicLLMCompletion(model_name=self.model, temperature=self.temperature,
                                                    max_tokens=self.max_tokens, **self.kwargs)
        elif self.api == 'together' and self.chat:
            self.api_model = ChatTogether(model=self.model, temperature=self.temperature,
                                          timeout=self.timeout, max_tokens=self.max_tokens, 
                                          api_key=os.getenv('TOGETHER_API_KEY'), 
                                          **self.kwargs)
        elif self.api == 'together' and not self.chat:
            self.api_model = TogetherLLMCompletion(model=self.model, temperature=self.temperature,
                                                    max_tokens=self.max_tokens, **self.kwargs)
        elif self.api == 'google' and self.chat:
            self.api_model = ChatGoogleGenerativeAI(model=self.model, temperature=self.temperature,
                                                    timeout=self.timeout, max_tokens=self.max_tokens, 
                                                    api_key=os.getenv('GOOGLE_API_KEY'), 
                                                    stream_usage=True, **self.kwargs)
        elif self.api == 'huggingface':
            self.tokenizer = AutoTokenizer.from_pretrained(self.model, padding_side='left', trust_remote_code=True)
            try:
                model = AutoModelForCausalLM.from_pretrained(self.model, trust_remote_code=True)
            except:
                model = AutoModelForCausalLM.from_pretrained(self.model, trust_remote_code=False, 
                                                             attn_implementation="eager")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model.generation_config.eos_token_id = self.tokenizer.eos_token_id
            model.config.eos_token_id = self.tokenizer.eos_token_id
            model.to(device)
            kwargs = self.kwargs.copy()
            if 'sequential' in kwargs:
                kwargs.pop('sequential')
            if "echo" in kwargs:
                kwargs.pop("echo")
            if "logprobs" in kwargs:
                kwargs.pop("logprobs")
            pipeline_chat = LogProbsTextGenerationPipeline(
                                model=model,
                                tokenizer=self.tokenizer, 
                                device='cuda', 
                                max_new_tokens=self.max_tokens,
                                do_sample=True if self.temperature > 0 else False,
                                temperature=self.temperature,
                                return_full_text=False,
                                **kwargs
                            )
            self.api_model = HuggingFacePipelineFixed(pipeline=pipeline_chat)
        else:
            raise ValueError(f'API {self.api} not supported or chat {self.chat} not supported')
        
    async def run_queries(self, queries):
        """
        Run queries against the API model.

        Args:
            queries (list): A list of queries to be executed. 
                            If chat is enabled, each query is a list of tuples, where each tuple contains ('system', 'ai', 'human') and the query message.
                            If chat is disabled, each query is a string.
        Returns:
            tuple: A tuple containing the outputs of the queries, the detailed cost, and the total cost.
            Total cost is a dictionary containing the input tokens, output tokens, and the total cost.
            Detailed cost is a list of dictionaries containing the same information for each query.

        Raises:
            ValueError: If the query type is not supported.
        """
        retry_api_model = self.api_model
        if self.chat:
            queries_converted = []
            if self.api == 'huggingface':
                for query in queries:
                    current_query = []
                    for query_type, query_message in query:
                        if query_type == 'system':
                            current_query.append({
                                'role': 'system',
                                'content': query_message
                            })
                        elif query_type == 'ai':
                            current_query.append({
                                'role': 'assistant',
                                'content': query_message
                            })
                        elif query_type == 'human':
                            current_query.append({
                                'role': 'user',
                                'content': query_message
                            })
                    current_query = self.tokenizer.apply_chat_template(current_query, tokenize=False)
                    queries_converted.append(current_query)
            else:
                for query in queries:
                    current_query = []
                    for query_type, query_message in query:
                        if query_type == 'system':
                            current_query.append(SystemMessage(content=query_message))
                        elif query_type == 'ai':
                            current_query.append(AIMessage(content=query_message))
                        elif query_type == 'human':
                            current_query.append(HumanMessage(content=query_message))
                        else:
                            raise ValueError(f'Query type {query_type} not supported')
                    queries_converted.append(current_query)
        else:
            queries_converted = queries

        if self.sequential:
            results, times = await self.run_sequential_queries(retry_api_model.abatch,  
                                                               queries_converted)
        else:
            results = await self._run_with_rate_limiting(retry_api_model.abatch, 
                                                     queries_converted)
            times = [0] * len(results)
        results = self.unify_output_format(results)
        cost, detailed_cost = self.get_cost(results)
        detailed_cost = [{'time': time, **detailed} for time, detailed in zip(times, detailed_cost)]
        outputs = [result['content'] for result in results]
        if self.return_logprobs:
            logprob_info = self.get_logprobs(results)
            outputs = [(result['content'], logprob) 
                       for result, logprob in zip(results, logprob_info)]
        return outputs, detailed_cost, cost
    
    def unify_output_format(self, results):
        """
        Unifies the output format of the given results across all APIs.

        Args:
            results (list): A list of results.

        Returns:
            list: A list of unified results.

        """
        unified_results = []
        for result in results:
            if result is None:
                unified_results.append({
                    'content': '',
                    'usage_metadata': {
                        'input_tokens': 0,
                        'output_tokens': 0
                    },
                    "response_metadata": {
                        "logprobs": {
                            "tokens": [],
                            "token_logprobs": []
                        }
                    }
                })
                continue
            if not isinstance(result, dict):
                result = dict(result)
            if 'generation_info' in result:
                result = result['generation_info']
            unified_results.append(result)
        return unified_results
    
    def get_logprobs(self, results):
        """
        Retrieves the log probabilities from the given results.
        Parameters:
            results (list): A list of results.
        Returns:
            list: A nested list containing the log probabilities for each result.
        Raises:
            None
        """
        if self.api == 'huggingface':
            logprob_info = []
            
            for result in results:
                logprob_info.append((
                    result['tokens'], 
                    result['logprobs']
                ))
            logprob_info = [
                [[(token, logprob)] for token, logprob in zip(result[0], result[1])]
                for result in logprob_info
            ]
        if self.api == 'together':
            logprob_info = []
            
            for result in results:
                logprob_info.append((
                    result['response_metadata']['logprobs']['tokens'], 
                    result['response_metadata']['logprobs']['token_logprobs']
                ))
            logprob_info = [
                [[(token, logprob)] for token, logprob in zip(result[0], result[1])]
                for result in logprob_info
            ]
        elif self.api == 'openai':
            logprob_info = []
            for result in results:
                result_specific_logprob = []
                for token_logprobs in result['response_metadata']['logprobs']['content']:
                    top_logprobs_token = []
                    for token in token_logprobs['top_logprobs']:
                        top_logprobs_token.append((token['token'], token['logprob']))

                    result_specific_logprob.append(top_logprobs_token)
                logprob_info.append(result_specific_logprob)
        return logprob_info

    async def run_sequential_queries(self, func, queries):
        """
        Run queries sequentially against the API model.

        Args:
            func (callable): The function to be executed.
            queries (list): A list of queries to be executed
        Returns:
            results and times
        """
        results = []
        times = []
        for query in tqdm(queries):
            time_start = time.time()
            for _ in range(self.max_retries):
                try:
                    result = await func([query])
                    break
                except Exception as e:
                    print('Traceback: ', e)
                    continue
            time_end = time.time()
            results.extend(result)
            times.append(time_end - time_start)
        return results, times

    async def _run_with_rate_limiting(self, func, queries):
        """
        Runs the given function with rate limiting.

        Args:
            func (callable): The function to be executed.
            queries (list): The list of queries to be processed.

        Returns:
            list: The results of the function execution.
        """
        results = []
        for i in tqdm(range(0, len(queries), self.rate_limiter.max_bucket_size), 
                      desc='Running queries'):
            batch = queries[i:i + self.rate_limiter.max_bucket_size]
            if self.api != 'huggingface':
                while not self.rate_limiter.acquire():
                    continue
            n_retries = 0
            for i in range(self.max_retries):
                try:
                    results.extend(await func(batch))
                    break
                except Exception as e:
                    print(f'Error: {e}')
                    n_retries += 1
                    continue
            if n_retries == self.max_retries:
                print('Failed to get response after max retries')
                results.extend([None] * len(batch))
        return results

    def get_cost(self, results):
        """
        Calculates the cost of the given results.

        Args:
            results (list): A list of results.

        Returns:
            tuple: A tuple containing the total input tokens, total output tokens, total cost, and detailed cost for each result.

        """
        input_tokens = 0
        output_tokens = 0
        detailed_cost = []
        for result in results:
            input_tokens += result['usage_metadata']['input_tokens']
            output_tokens += result['usage_metadata']['output_tokens']
            detailed = result['usage_metadata']
            detailed['cost'] = detailed['input_tokens'] * self.read_cost / 10 ** 6 
            detailed['cost'] += detailed['output_tokens'] * self.write_cost / 10 ** 6
            detailed_cost.append(detailed)

        return {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'cost': input_tokens * self.read_cost / 10 ** 6 + output_tokens * self.write_cost / 10 ** 6
        }, detailed_cost