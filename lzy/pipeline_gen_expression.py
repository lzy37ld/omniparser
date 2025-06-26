from pydantic import BaseModel
from openai import OpenAI, AsyncOpenAI, DefaultAioHttpClient
import base64
import os
import json
import argparse
import asyncio
from typing import List
import time
import backoff
import openai
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm

context_prompt = """
You are given an screenshot input. Your task is to generate natural language referring expressions which specify different target text spans contained within the screenshot that human tend to use mouse drag action to select. Ignore the parts that are not text, that are not selectable by mouse and that are not the places where human tend to select in daily life.

{category_prompt}

The referring expression should be clear about the granularity of the text, i.e., clearly specify if they are pargagraph(s), line(s), sentence(s), words without using ambiguous words like 'text', 'part'. The target text span can be single or multiple paragraphs, lines, sentences. For words, it should be at least multiple words as selecting a single word usually does not require a mouse drag action.

If no feasible or available referring expression meeting the requirements can be generated, you should return False for availability.
If it does, you should return True for availability and the generated referring expressions.
"""

category_prompt_semantic = """
For the referring expression you generated, they must describe the target text span based on its meaning, intent, or topical content.

For example:
a.Select the paragraph discussing how to download models.
b.Select the lines that infer the causes of failure.
c.Select the sentence about Kobe Bryant's career.
d.Select consecutive words referring to the weight of the MacBook Pro.
"""

category_prompt_positional = """
For the referring expression you generated, they must refer to the positional location of the text—either in absolute terms (e.g., top, bottom of the page) or relative to other (visual) elements.

For example:
 a. Select the second last paragraph at the bottom of the page.
 b. Select the first three lines, exact under the code block.
 c. Select the sentence immediately below the chart title.
 d. Select the words on the left side of the login button.
"""

category_prompt_lexical = """
For the referring expression you generated, they must describe the text by referencing its literal or quoted content, including the starting words, key phrases, or exact match.

For example:
 a. Select the paragraph that begins with "To get started with Python…".
 b. Select the lines ending with "before submission is due".
 c. Select the sentence containing the phrase "AI is transforming industries".
 d. Select the words that say "Monday, Tuesday, and so on".
"""

category_prompt_visual = """
For the referring expression you generated, they must refer to distinctive visual features of the text, such as font color, size, emphasis, or highlighting.

For example:
 a. Select the paragraph written in bold italics.
 b. Select the lines highlighted in yellow.
 c. Select the sentence in red font.
 d. Select the words with the largest font size on the screen.
"""

class Output_gen_expression(BaseModel):
    availability: bool
    expressions: list[str]

system_prompt_map = {
    "semantic": category_prompt_semantic,
    "positional": category_prompt_positional,
    "lexical": category_prompt_lexical,
    "visual": category_prompt_visual
}

output_map = {
    "all": None,
    "semantic": Output_gen_expression,
    "positional": Output_gen_expression,
    "lexical": Output_gen_expression,
    "visual": Output_gen_expression
}

def encode_image(image_path):
    """Encode image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

class LLMClient:
    """Synchronous OpenAI client"""
    def __init__(self, model, api_key=None):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    @backoff.on_exception(
        backoff.expo,
        (openai.RateLimitError, openai.APIError, openai.APIConnectionError),
        max_time=300,
        max_tries=6
    )
    def call_llm(self, system_prompt, input_text, image_path, response_format=None):
        """Call OpenAI API with structured output support"""
        base64_image = encode_image(image_path)

        messages = [
            {
                "role": "developer",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": input_text,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        }
                    }
                ]
            }
        ]

        if response_format:
            # Use structured output
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=response_format
            )
        else:
            # Use regular chat completion
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
        
        return response

class AsyncLLMClient:
    """Asynchronous OpenAI client with aiohttp for better concurrency performance"""
    def __init__(self, model, api_key=None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = None

    async def __aenter__(self):
        """Async context manager entry"""
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            http_client=DefaultAioHttpClient()  # Use aiohttp for better performance
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.client:
            await self.client.close()

    @backoff.on_exception(
        backoff.expo,
        (openai.RateLimitError, openai.APIError, openai.APIConnectionError),
        max_time=300,
        max_tries=6
    )
    async def call_llm_async(self, system_prompt, input_text, image_path, response_format=None):
        """Async call to OpenAI API with structured output support"""
        if not self.client:
            raise RuntimeError("AsyncLLMClient must be used within async context manager")
            
        base64_image = encode_image(image_path)

        messages = [
            {
                "role": "developer",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": input_text,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        }
                    }
                ]
            }
        ]

        if response_format:
            # Use structured output
            response = await self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=response_format
            )
        else:
            # Use regular chat completion
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
        
        return response

async def generate_referring_expressions_async(image_path, category, save_dir, llm_client, pbar=None):
    """Async version of generate_referring_expressions with automatic retry"""
    try:
        input_text = "Here is the screenshot."
        system_prompt = context_prompt.format(
            category_prompt=system_prompt_map[category].strip()
        ).strip()
        
        # Call API
        response = await llm_client.call_llm_async(
            system_prompt=system_prompt,
            input_text=input_text,
            image_path=image_path,
            response_format=output_map[category]
        )
        
        # Extract content
        if output_map[category]:  # Structured output
            expressions = dict(response.choices[0].message.parsed)
        else:  # Regular output
            expressions = response.choices[0].message.content
            if isinstance(expressions, str):
                try:
                    expressions = json.loads(expressions)
                except json.JSONDecodeError:
                    expressions = {"content": expressions}
        
        # Save results
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(save_dir, f'gen_model-{llm_client.model}', f'category-{category}')
        os.makedirs(save_path, exist_ok=True)
        
        file_path = os.path.join(save_path, f'name-{image_name}.json')
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump({"expressions": expressions}, f, indent=4, ensure_ascii=False)
        
        # Update progress bar
        if pbar:
            pbar.set_postfix_str(f"✓ {image_name}")
            pbar.update(1)
        
        return True
        
    except Exception as e:
        image_name = os.path.basename(image_path)
        error_msg = f"✗ {image_name}: {str(e)[:50]}..."
        
        # Update progress bar with error
        if pbar:
            pbar.set_postfix_str(error_msg)
            pbar.update(1)
        else:
            print(f"Error processing {image_name}: {str(e)}")
        
        return False

async def process_images_concurrently(image_list, category, save_dir, model, api_key, max_concurrent=10):
    """Process images concurrently with rate limiting using aiohttp for better performance"""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Create progress bar
    pbar = async_tqdm(
        total=len(image_list),
        desc=f"Processing {category} expressions",
        unit="img",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
    )
    
    async def process_with_semaphore(image_path, llm_client):
        async with semaphore:
            # Add small delay to avoid rate limiting
            await asyncio.sleep(0.1)
            return await generate_referring_expressions_async(
                image_path, category, save_dir, llm_client, pbar
            )
    
    try:
        # Use async context manager for optimal connection management
        async with AsyncLLMClient(model, api_key) as llm_client:
            # Create tasks
            tasks = [process_with_semaphore(image_path, llm_client) for image_path in image_list]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count results
            success_count = sum(1 for result in results if result is True)
            failure_count = len(results) - success_count
            
            # Final summary
            pbar.set_postfix_str(f"Complete! ✓{success_count} ✗{failure_count}")
            
    finally:
        pbar.close()
    
    print(f"\n🎯 Processing Summary:")
    print(f"✅ Success: {success_count}")
    print(f"❌ Failed: {failure_count}")
    print(f"📊 Total: {len(results)}")
    print(f"📈 Success Rate: {success_count/len(results)*100:.1f}%")
    
    return results
def generate_referring_expressions(image_path, category, save_dir, llm_client):
    """Synchronous version of generate_referring_expressions"""
    input_text = "Here is the screenshot."
    system_prompt = context_prompt.format(
        category_prompt=system_prompt_map[category].strip()
    ).strip()

    # Call API
    response = llm_client.call_llm(
        input_text=input_text,
        image_path=image_path,
        system_prompt=system_prompt,
        response_format=output_map[category]
    )

    # Extract content
    if output_map[category]:  # Structured output
        expressions = dict(response.choices[0].message.parsed)
    else:  # Regular output
        expressions = response.choices[0].message.content
        if isinstance(expressions, str):
            try:
                expressions = json.loads(expressions)
            except json.JSONDecodeError:
                expressions = {"content": expressions}

    # Save results
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(save_dir, f'gen_model-{llm_client.model}', f'category-{category}')
    os.makedirs(save_path, exist_ok=True)
    
    file_path = os.path.join(save_path, f'name-{image_name}.json')
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump({"expressions": expressions}, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="/home/t-zeyiliao/OmniParser/lzy_images/test.png")
    parser.add_argument("--image_list_from_mapping_dict", type=str, default="/home/t-zeyiliao/OmniParser/parsed_results/screenspot_pro/parsed_mode-line-mapping_dict.json")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--category", type=str, default="positional")
    parser.add_argument("--save_dir", type=str, default="/home/t-zeyiliao/OmniParser/referring_expressions/screenspot_pro")
    parser.add_argument("--max_concurrent", type=int, default=30, help="Maximum number of concurrent API calls")
    parser.add_argument("--use_async", action="store_true", help="Use async processing instead of sequential")
    
    args = parser.parse_args()

    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        exit(1)

    # Load image list
    with open(args.image_list_from_mapping_dict, 'r', encoding='utf-8') as f:
        image_list_from_mapping_dict = json.load(f)

    image_list = list(image_list_from_mapping_dict["original_to_output"].keys())
    
    print(f"📋 Configuration:")
    print(f"   • Images: {len(image_list)}")
    print(f"   • Mode: {'Async (aiohttp)' if args.use_async else 'Sequential'}")
    print(f"   • Category: {args.category}")
    print(f"   • Model: {args.model}")
    if args.use_async:
        print(f"   • Max concurrent: {args.max_concurrent}")
    print(f"   • Auto retry: Up to 6 attempts with exponential backoff")
    print()
    
    if args.use_async:
        # Run async processing with aiohttp for better performance
        asyncio.run(process_images_concurrently(
            image_list, args.category, args.save_dir, args.model, api_key, args.max_concurrent
        ))
    else:
        # Create sync client
        sync_llm_client = LLMClient(args.model, api_key)
        
        # Sequential processing with progress bar
        success_count = 0
        failure_count = 0
        
        with tqdm(image_list, desc=f"Processing {args.category} expressions", unit="img", ncols=100) as pbar:
            for image_path in pbar:
                try:
                    image_name = os.path.basename(image_path)
                    pbar.set_postfix_str(f"Processing {image_name[:30]}...")
                    
                    generate_referring_expressions(
                        image_path, args.category, save_dir=args.save_dir, llm_client=sync_llm_client
                    )
                    
                    success_count += 1
                    pbar.set_postfix_str(f"✓ {image_name}")
                    
                except Exception as e:
                    failure_count += 1
                    pbar.set_postfix_str(f"✗ {image_name}: {str(e)[:30]}...")
        
        print(f"\n🎯 Processing Summary:")
        print(f"✅ Success: {success_count}")
        print(f"❌ Failed: {failure_count}")
        print(f"📊 Total: {len(image_list)}")
        print(f"📈 Success Rate: {success_count/len(image_list)*100:.1f}%")