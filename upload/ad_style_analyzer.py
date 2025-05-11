import os
import json
import logging
from pathlib import Path
from google import genai
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

class SimilarExamples(BaseModel):
    examples: List[str]
    indices: List[int]
    
# Add this class definition after the SimilarExamples class
class PunctuationStyle(BaseModel):
    emojis: str
    exclamation: str
    questions: str

class StyleGuide(BaseModel):
    tone: str
    pacing: str
    vocabulary: List[str]
    punctuation: PunctuationStyle
    structure: str
    common_phrases: Optional[List[str]] = None
    hooks: Optional[List[str]] = None
    calls_to_action: Optional[List[str]] = None
    
class AdStyleAnalyzer:
    """
    A class to extract text from UGC content and analyze its style using Google's Gemini model.
    """
    
    def __init__(self, api_key=None):
        """
        Initialize the UGCStyleAnalyzer with Google API key.
        
        Args:
            api_key (str, optional): Google API key. If not provided, will try to get from environment variable.
        """
        # Get API key from parameter or environment variable
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        
        if not self.api_key:
            logger.error("No Google API key provided. Please set GOOGLE_API_KEY environment variable or pass it to the constructor.")
            raise ValueError("Google API key is required")
        
        self.client = genai.Client(api_key=self.api_key)
        
    def extract_text_from_json(self, json_path: str = None, username: str = None) -> List[str]:
        """
        Extract text fields from Instagram UGC content by calling the API multiple times.
        
        Args:
            json_path (str, optional): Path to a local JSON file for testing. If not provided, 
                                      will call the Instagram API directly.
            username (str, optional): Instagram username to analyze. If not provided, will use
                                     the value from environment variable or default to "ugcwithkrystle".
            
        Returns:
            List[str]: List of extracted text content
        """
        import requests
        
        texts = []
        
        # If a local JSON file is provided, use it instead of API
        if json_path and os.path.exists(json_path):
            logger.info(f"Using local JSON file: {json_path}")
            try:
                # Load the JSON file
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract text from captions
                if 'items' in data and isinstance(data['items'], list):
                    for item in data['items']:
                        if 'caption' in item and 'text' in item['caption']:
                            texts.append(item['caption']['text'])
                
                logger.info(f"Extracted {len(texts)} text entries from local JSON file")
                return texts
                
            except Exception as e:
                logger.error(f"Error extracting text from local JSON: {str(e)}", exc_info=True)
                raise RuntimeError(f"Failed to extract text from local JSON: {str(e)}")
        
        # Get username from parameter, environment variable, or default
        instagram_username = username or os.getenv("STYLE_INSTAGRAM_USERNAME", "ugcwithkrystle")
        
        # Resolve user_id from username
        try:
            logger.info(f"Resolving user ID for Instagram username: {instagram_username}")
            id_url = "https://instagram-looter2.p.rapidapi.com/id"
            id_querystring = {"username": instagram_username}
            
            headers = {
                "x-rapidapi-key": os.getenv("RAPIDAPI_KEY"),
                "x-rapidapi-host": "instagram-looter2.p.rapidapi.com"
            }
            
            id_response = requests.get(id_url, headers=headers, params=id_querystring)
            id_response.raise_for_status()
            
            id_data = id_response.json()
            
            if not id_data.get("status", False) or "user_id" not in id_data:
                logger.error(f"Failed to resolve user ID for {instagram_username}: {id_data}")
                raise RuntimeError(f"Failed to resolve user ID for {instagram_username}")
            
            user_id = id_data["user_id"]
            logger.info(f"Resolved user ID for {instagram_username}: {user_id}")
            
        except Exception as e:
            logger.error(f"Error resolving Instagram user ID: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to resolve Instagram user ID: {str(e)}")
        
        # API parameters for user feeds
        url = "https://instagram-looter2.p.rapidapi.com/user-feeds"
        count = "12"
        
        # Initialize variables for API loop
        max_calls = 10
        call_count = 0
        more_available = True
        next_max_id = None
        all_items = []
        
        logger.info(f"Starting API calls to fetch Instagram content for user {user_id}")
        
        # Rest of the function remains the same
        # Loop until we reach max_calls or no more content is available
        while more_available and call_count < max_calls:
            # Prepare query parameters
            querystring = {"id": user_id, "count": count}
            
            # Add next_max_id if we have it from previous call
            if next_max_id:
                querystring["max_id"] = next_max_id
            
            try:
                # Make API call
                logger.info(f"Making API call {call_count + 1}/{max_calls} with params: {querystring}")
                response = requests.get(url, headers=headers, params=querystring)
                response.raise_for_status()  # Raise exception for HTTP errors
                
                # Parse response
                data = response.json()
                
                # Check if the response has the expected structure
                if 'items' in data and isinstance(data['items'], list):
                    # Add items to our collection
                    all_items.extend(data['items'])
                    logger.info(f"Received {len(data['items'])} items in this batch")
                    
                    # Extract text from captions in this batch
                    for item in data['items']:
                        if 'caption' in item and 'text' in item['caption']:
                            texts.append(item['caption']['text'])
                
                # Check if more content is available
                more_available = data.get('more_available', False)
                
                # Get next_max_id for pagination if available
                next_max_id = data.get('next_max_id')
                
                # If no next_max_id but more_available is True, something is wrong
                if more_available and not next_max_id:
                    logger.warning("more_available is True but no next_max_id provided")
                    break
                
                # Increment call counter
                call_count += 1
                
                # Add a small delay to avoid rate limiting
                if more_available and call_count < max_calls:
                    import time
                    time.sleep(1)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"API request failed: {str(e)}", exc_info=True)
                raise RuntimeError(f"Failed to fetch Instagram content: {str(e)}")
            except Exception as e:
                logger.error(f"Error processing API response: {str(e)}", exc_info=True)
                raise RuntimeError(f"Failed to process Instagram content: {str(e)}")
        
        logger.info(f"Completed {call_count} API calls, extracted {len(texts)} text entries")
        
        # Save the combined results to a file for future reference
        try:
            output_dir = os.path.dirname(__file__)
            combined_json_path = os.path.join(output_dir, "combined_instagram_data.json")
            with open(combined_json_path, 'w', encoding='utf-8') as f:
                json.dump({"items": all_items, "more_available": more_available}, f, indent=2)
            logger.info(f"Saved combined data to {combined_json_path}")
        except Exception as e:
            logger.warning(f"Failed to save combined data: {str(e)}")
        
        return texts
    
    
    
    # Then modify the analyze_style method
    def analyze_style(self, texts: List[str]) -> bool:
        """
        Analyze the style and tone of the provided texts using Gemini.
        
        Args:
            texts (List[str]): List of text content to analyze
            
        Returns:
            bool: True if analysis was successful and saved to file, False otherwise
        """
        if not texts:
            logger.warning("No texts provided for style analysis")
            return False
        
        logger.info(f"Analyzing style of {len(texts)} text entries")
        
        # Combine texts for analysis with clear separators
        combined_text = "\n\n---\n\n".join(texts)
        
        # Create the prompt for style analysis based on the workflow
        prompt = f"""
        Here are several summaries/captions from a content creator. Analyze them and list:
        
        • Tone (e.g. casual, playful, urgent)
        • Sentence length & structure
        • Favorite words, emojis, punctuation quirks
        • Typical opening/closing hooks
        • Use of questions or calls-to-action
        
        Format your response as a JSON object with these exact keys and types:
        - tone: string
        - pacing: string
        - vocabulary: array of strings
        - punctuation: object with exactly these keys (all values must be strings):
            - emojis
            - exclamation
            - questions
        - structure: string
        - common_phrases: array of strings
        - hooks: array of strings
        - calls_to_action: array of strings
        
        Do not include any other keys or types. All values must be strings or arrays of strings as specified.

        TEXTS TO ANALYZE:
        {combined_text}
        """
        
        try:
            # Generate content analysis with structured output
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': StyleGuide,
                }
            )
            
            # Parse the response into a StyleGuide object
            analysis = response.parsed
            
            # Save both the analysis and the original texts to a single JSON file
            output_dir = os.path.dirname(__file__)
            combined_data_path = os.path.join(output_dir, "style_analysis.json")
            
            # Create a dictionary with both the style guide and the texts
            combined_data = {
                "style_guide": analysis.model_dump(),
                "example_texts": texts
            }
            
            # Save to JSON file
            with open(combined_data_path, "w", encoding="utf-8") as f:
                json.dump(combined_data, f, indent=2)
                
            logger.info(f"Style analysis and example texts saved to {combined_data_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error analyzing style: {str(e)}", exc_info=True)
            return False
    
    def select_similar_examples(self, texts: List[str], new_content: str, num_examples: int = 3) -> List[str]:
        """
        Select the most topically similar examples to the new content using vector embeddings.
        
        Args:
            texts (List[str]): List of existing text content
            new_content (str): New content to compare against
            num_examples (int): Number of examples to select
            
        Returns:
            List[str]: List of selected examples
        """
        if not texts or len(texts) <= num_examples:
            return texts
        
        logger.info(f"Selecting {num_examples} most similar examples from {len(texts)} texts using embeddings")
        
        try:
            import numpy as np
            import faiss
            
            # Initialize Gemini client
            client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
            
            # Get embedding for the new content
            new_content_result = client.models.embed_content(
                model=os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004"),
                contents=new_content
            )
            # Extract the values from the embeddings
            new_content_embedding = np.array([new_content_result.embeddings[0].values], dtype=np.float32)
            
            # Get embeddings for all texts
            all_embeddings = []
            for text in texts:
                result = client.models.embed_content(
                    model=os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004"),
                    contents=text
                )
                all_embeddings.append(result.embeddings[0].values)
            
            # Convert to numpy array
            all_embeddings = np.array(all_embeddings, dtype=np.float32)
            
            # Create FAISS index - using L2 distance (Euclidean)
            dimension = len(new_content_embedding[0])
            index = faiss.IndexFlatL2(dimension)
            
            # Add embeddings to the index
            index.add(all_embeddings)
            
            # Search for the most similar texts
            distances, indices = index.search(new_content_embedding, min(num_examples, len(texts)))
            
            # Get the indices of the most similar texts
            similar_indices = indices[0].tolist()
            
            logger.info(f"Found {len(similar_indices)} similar examples with indices: {similar_indices}")
            
            # Return the selected examples
            return [texts[i] for i in similar_indices]
            
        except Exception as e:
            logger.error(f"Error selecting similar examples with embeddings: {str(e)}", exc_info=True)
            # Fallback to first N examples
            logger.warning(f"Falling back to first {num_examples} examples")
            return texts[:min(num_examples, len(texts))]
    
    def generate_content_in_style(self, style_data: Dict[str, Any], content_to_style: str) -> str:
        """
        Generate new content based on the analyzed style.
        
        Args:
            style_data (Dict[str, Any]): Style guide data containing style analysis and examples
            content_to_style (str): Content to be styled (factual summary)
            
        Returns:
            str: Generated content in the analyzed style
        """
        logger.info("Generating content based on style analysis")
        
        try:
            # Try to parse as StyleGuide
            try:
                # Extract the style_guide object from the combined data
                style_guide_data = style_data.get('style_guide', {})
                style_guide_obj = StyleGuide.model_validate(style_guide_data)
                
                # Format the style guide as text
                punctuation = style_guide_obj.punctuation
                style_guide = f"""
                  Tone: {style_guide_obj.tone}
                  Pacing: {style_guide_obj.pacing}
                  Vocabulary: {', '.join(style_guide_obj.vocabulary)}
                  Punctuation:
                  - Emojis: {punctuation.emojis}
                  - Exclamation: {punctuation.exclamation}
                  - Questions: {punctuation.questions}
                  Structure: {style_guide_obj.structure}
                  Common Phrases: {', '.join(style_guide_obj.common_phrases or [])}
                  Hooks: {', '.join(style_guide_obj.hooks or [])}
                  Calls to Action: {', '.join(style_guide_obj.calls_to_action or [])}
                  """
            except Exception as e:
                logger.warning(f"Could not parse JSON as StyleGuide: {str(e)}")
                return f"Error parsing style guide: {str(e)}"
        
            # Get examples from style_data
            examples = style_data.get("example_texts", [])
            
            # Format examples if available
            examples_text = ""
            if examples:
                examples_text = "\n\nExamples of content in this style:\n"
                for i, example in enumerate(examples):
                    examples_text += f"\nExample {i+1}:\n{example}\n"
            # Create the prompt for content generation using the few-shot approach
            generation_prompt = f"""
            You are a video-summary assistant for {os.getenv("TIKTOK_ACCOUNT")} ({os.getenv("EMAIL")}).
            
            Follow this style guide EXACTLY:
            {style_guide}
            
            Examples of similar summaries:
            {examples_text}
            
            Now rewrite the following factual content in the same style as the examples and following the style guide:
            
            "{content_to_style}"
            
            Make sure your response maintains all the factual information while matching the tone, structure, and style patterns.
            
            DO NOT include the account or email from {examples_text}
            DO NOT include any hashtags
            DO NOT include If you are a brand reading this...
            """
            
            # Log environment variables for contact info
            logger.info(f"Contact Info Environment Variables - Email: {os.getenv('EMAIL')}, TikTok: {os.getenv('TIKTOK_ACCOUNT')}, Instagram: {os.getenv('INSTAGRAM_ACCOUNT')}")
            
            # Generate content
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=generation_prompt,
            )
            
            # Add contact information after the generated content
            contact_info = f"""

If you're a brand reading this and want to collaborate with a UGC content creator, let's chat! 👋 Let's create something amazing together!

Send me a DM or email me, I'd love to hear from you! Down here ⬇️
Email: {os.getenv("EMAIL")}
TikTok: {os.getenv("TIKTOK_ACCOUNT")}
Instagram: {os.getenv("INSTAGRAM_ACCOUNT")}"""

            # Return the generated text with contact info appended
            return response.text + contact_info
            
        except Exception as e:
            logger.error(f"Error generating content: {str(e)}", exc_info=True)
            return f"Error generating content: {str(e)}"
    
    def apply_style_to_video_analysis(self, 
                                     factual_summary: str, 
                                     factual_title: str, 
                                     style_analysis_path: str,
                                     use_similar_examples: bool = True) -> Dict[str, str]:
        """
        Apply the analyzed style to video analysis results.
        
        Args:
            factual_summary (str): Factual summary from video analysis
            factual_title (str): Factual title from video analysis
            style_analysis_path (str): Path to the JSON file containing style analysis and example texts
            use_similar_examples (bool): Whether to use similar examples for styling (default: True)
            
        Returns:
            Dict[str, str]: Styled summary and title
        """
        logger.info(f"Applying style to video analysis from: {style_analysis_path}")
        
        # Load style analysis and example texts from JSON file
        try:
            with open(style_analysis_path, 'r', encoding='utf-8') as f:
                style_data = json.load(f)
            
            if use_similar_examples:
                # Get all examples
                example_texts = style_data.get("example_texts", [])
                if example_texts:
                    # Find similar examples
                    similar_examples = self.select_similar_examples(example_texts, factual_summary)
                    # Update the examples in style_data
                    style_data["example_texts"] = similar_examples
                    logger.info(f"Selected {len(similar_examples)} similar examples for styling")
                else:
                    logger.warning("No example texts found in the style analysis file")
            
        except Exception as e:
            logger.error(f"Error loading style analysis file: {str(e)}", exc_info=True)
            return {
                "styled_summary": f"Error: {str(e)}",
                "styled_title": f"Error: {str(e)}",
                "original_summary": factual_summary,
                "original_title": factual_title
            }
        
        # Generate styled summary and title
        styled_summary = self.generate_content_in_style(style_data, factual_summary)
        styled_title = self.generate_content_in_style(style_data, factual_title)
        
        return {
            "styled_summary": styled_summary,
            "styled_title": styled_title,
            "original_summary": factual_summary,
            "original_title": factual_title
        }


def main():
    """
    Main function to demonstrate the UGCStyleAnalyzer.
    """
    try:
        # Initialize the analyzer
        analyzer = AdStyleAnalyzer()
        
        # Path to the JSON file
        json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "title_template.json")
        
        # Extract text from JSON using the local file
        json_path = os.path.join(os.path.dirname(__file__), "combined_instagram_data.json")
        texts = analyzer.extract_text_from_json(json_path)
        
        if not texts:
            logger.warning("No texts found in the JSON file")
            return
        
        # Analyze style
        style_analysis = analyzer.analyze_style(texts)
        
        # Generate a sample content based on the style
        sample_prompt = "A new skincare product that helps with hydration"
        style_analysis_path = os.path.join(os.path.dirname(__file__), "style_analysis.json")
        styled_content = analyzer.apply_style_to_video_analysis(
            factual_summary=sample_prompt,
            factual_title="",
            style_analysis_path=style_analysis_path
        )
        generated_content = styled_content["styled_summary"]
        
        # Save the generated content to a file
        output_path = os.path.join(os.path.dirname(__file__), "generated_sample.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(generated_content)
            
        logger.info(f"Generated content saved to {output_path}")
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()