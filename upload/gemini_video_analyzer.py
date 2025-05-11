import os
import logging
from pickle import TRUE
from google import genai
import time
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from pathlib import Path

from upload.ad_style_analyzer import AdStyleAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Define Pydantic models for structured output
class Hashtag(BaseModel):
    tag: str
    category: Optional[str] = None
    relevance: Optional[str] = None

class VideoAnalysis(BaseModel):
    title: str
    summary: str
    hashtags: List[Hashtag]
    key_topics: Optional[List[str]] = None


class GeminiVideoAnalyzer:
    """
    A class to analyze videos using Google's Gemini 1.5 Flash model.
    """
    
    def __init__(self, api_key=None, style_analyzer=None):
        """
        Initialize the GeminiVideoAnalyzer with Google API key.
        
        Args:
            api_key (str, optional): Google API key. If not provided, will try to get from environment variable.
            style_analyzer (AdStyleAnalyzer, optional): Style analyzer for content styling. If not provided, a new one will be created when needed.
        """
        # Get API key from parameter or environment variable
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        
        if not self.api_key:
            logger.error("No Google API key provided. Please set GOOGLE_API_KEY environment variable or pass it to the constructor.")
            raise ValueError("Google API key is required")
        
        self.client = genai.Client(api_key=self.api_key)
        
        # Store the style analyzer or create it when needed
        self.style_analyzer = style_analyzer
    
    def analyze_video(self, video_path, prompt=None, apply_style=TRUE, style_json_path=None):
        """
        Analyze a video using Gemini 1.5 Flash model.
        This method uploads the video, waits for it to be processed, and then generates content.
        
        Args:
            video_path (str): Path to the video file
            prompt (str, optional): Custom prompt for analysis. If not provided, a default prompt will be used.
            apply_style (bool, optional): Whether to apply style to the title and summary. Default is False.
            style_json_path (str, optional): Path to the style analysis JSON file. Required if apply_style is True.
            
        Returns:
            VideoAnalysis: Structured analysis result with summary and hashtags
        """
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        logger.info(f"Analyzing video: {video_path}")
        
        video_file = None # Initialize video_file to None for potential cleanup in finally block
        try:
            # 1. Upload the video file
            logger.info(f"Uploading video file: {video_path}...")
            # Use the filename as the display name
            display_name = os.path.basename(video_path)
            video_file = self.client.files.upload(
                file=video_path,
                config={
                    "display_name": display_name,
                    "mime_type": "video/mp4"  # Assuming mp4, adjust if needed
                }
            )
            logger.info(f"Uploaded file '{video_file.display_name}' as: {video_file.name} (URI: {video_file.uri})")
            logger.info(f"Initial state: {video_file.state.name}")

            # 2. Wait for the file to be processed and become ACTIVE
            logger.info("Waiting for file to become ACTIVE...")
            while video_file.state.name == "PROCESSING":
                logger.info("File is still PROCESSING. Waiting 10 seconds...")
                time.sleep(10)  # Wait for 10 seconds
                video_file = self.client.files.get(name=video_file.name)  # Fetch the latest file state
                logger.info(f"Current state: {video_file.state.name}")

            if video_file.state.name == "ACTIVE":
                logger.info(f"File '{video_file.name}' is now ACTIVE and ready for use.")


                # Default prompt if none provided
                if not prompt:
                    prompt = """
                    Analyze this video and provide:
                    1. A concise summary of the content
                    2. A list of relevant hashtags for social media
                    
                    Format your response as structured data with a summary field and hashtags list.
                    For each hashtag, include the tag text and categorize it (general, niche, trending, etc.).
                    """
                
                # Create the prompt parts
                prompt_parts = [
                    prompt,
                    video_file  # Pass the File object directly
                ]

                logger.info("Generating content with structured output...")
                
                # Request structured output using the VideoAnalysis schema
                response = self.client.models.generate_content(
                    model = 'gemini-2.0-flash',
                    contents=prompt_parts,
                    config={
                        'response_mime_type': 'application/json',
                        'response_schema': VideoAnalysis,
                    }
                )
                
                # Parse the response into a VideoAnalysis object
                analysis = response.parsed
                logger.info(f"Generated analysis with {len(analysis.hashtags)} hashtags")
                
                # Apply style to title and summary if requested
                if apply_style:
                    logger.info("Applying style to title and summary...")
                                                # Initialize style analyzer if not provided
                    if self.style_analyzer is None:
                                logger.info("Creating new AdStyleAnalyzer instance")
                                self.style_analyzer = AdStyleAnalyzer(api_key=self.api_key)
                    # Ensure we have a valid style_json_path
                    if not style_json_path or not os.path.exists(style_json_path):
                        # Try to load from upload directory first
                        upload_dir_path = os.path.join(os.path.dirname(__file__), "style_analysis.json")
                        if os.path.exists(upload_dir_path):
                            style_json_path = upload_dir_path
                            logger.info(f"Found style_analysis.json in upload directory: {style_json_path}")
                        else:
                            # Create a new style analysis if path doesn't exist
                            logger.info("No existing style analysis found, creating new one...")
                            
                            # Extract text from JSON using the local file
                            texts = self.style_analyzer.extract_text_from_json()
                            
                            if not texts:
                                logger.warning("No texts found in the JSON file")
                                return analysis
                        
                            # Analyze style and save to file
                            if self.style_analyzer.analyze_style(texts):
                                style_json_path = os.path.join(os.path.dirname(__file__), "style_analysis.json")
                            else:
                                logger.info("Style analysis failed.")
                                return analysis
                        
                    # Apply style to the analysis using the updated method signature
                    styled_content = self.style_analyzer.apply_style_to_video_analysis(
                        factual_summary=analysis.summary,
                        factual_title=analysis.title,
                        style_analysis_path=style_json_path,
                        use_similar_examples=True  # Default to using similar examples
                    )
                    
                    # Update the analysis with styled content
                    analysis.title = styled_content.get("styled_title", analysis.title)
                    analysis.summary = styled_content.get("styled_summary", analysis.summary)
                    
                    # Add common UGC hashtags from style analysis
                    common_hashtags = [
                        Hashtag(tag="#ugccreator", category="general"),
                        Hashtag(tag="#ugccommunity", category="general"),
                        Hashtag(tag="#ugcaustralia", category="location"),
                        Hashtag(tag="#contentcreatoraustralia", category="niche"),
                        Hashtag(tag="#socialmediamarketing", category="industry"),
                        Hashtag(tag="#ugcmarketing", category="industry")
                    ]
                    analysis.hashtags.extend(common_hashtags)
                    
                    logger.info("Style applied successfully")
                
                return analysis

            elif video_file.state.name == "FAILED":
                error_message = f"File '{video_file.name}' processing FAILED."
                if hasattr(video_file, 'state_reason') and video_file.state_reason:
                    error_message += f" Reason: {video_file.state_reason}"
                logger.error(error_message)
                raise RuntimeError(error_message)
            else:
                error_message = f"File '{video_file.name}' is in an unexpected state: {video_file.state.name}"
                logger.error(error_message)
                raise RuntimeError(error_message)
            
        except Exception as e:
            logger.error(f"Error analyzing video: {str(e)}", exc_info=True)
            # Re-raise the exception or return a specific error message
            raise RuntimeError(f"Failed to analyze video: {str(e)}")
        finally:
            # Optional: Delete the file if it was uploaded and you no longer need it.
            # Be cautious with auto-deletion, ensure it's desired behavior.
            # if video_file:
            #     try:
            #         logger.info(f"Attempting to delete file {video_file.name}...")
            #         genai.delete_file(name=video_file.name)
            #         logger.info(f"File {video_file.name} deleted successfully.")
            #     except Exception as del_e:
            #         logger.error(f"Error deleting file {video_file.name}: {del_e}", exc_info=True)
            pass # No deletion by default for now

    def summarize_video(self, video_path, apply_style=False, style_json_path=None, example_texts=None):
        """
        Generate a summary of the video.
        
        Args:
            video_path (str): Path to the video file
            apply_style (bool, optional): Whether to apply style to the title and summary. Default is False.
            style_json_path (str, optional): Path to the style analysis JSON file. Required if apply_style is True.
            example_texts (List[str], optional): Example texts to use for style analysis.
            
        Returns:
            str: Summary text
        """
        prompt = "Provide a detailed summary of this video. Include key points, main topics, and important information."
        return self.analyze_video(video_path, prompt, apply_style, style_json_path, example_texts)

    def create_quiz(self, video_path, num_questions=5):
        """
        Create a quiz based on the video content.
        
        Args:
            video_path (str): Path to the video file
            num_questions (int): Number of questions to generate
            
        Returns:
            str: Quiz with answer key
        """
        prompt = f"Create a quiz with {num_questions} questions and an answer key based on the information in this video."
        return self.analyze_video(video_path, prompt)

    def extract_key_insights(self, video_path):
        """
        Extract key insights from the video.
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            str: Key insights text
        """
        prompt = "Extract the 5 most important insights from this video. For each insight, provide a brief explanation."
        return self.analyze_video(video_path, prompt)


    def get_hashtags(self, video_path):
        """
        Extract only hashtags from a video.
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            List[Hashtag]: List of hashtags
        """
        prompt = "Analyze this video and generate a list of 15-20 relevant hashtags for social media. For each hashtag, provide the tag text and categorize it as general, niche, or trending."
        analysis = self.analyze_video(video_path, prompt)
        return analysis.hashtags
    

# Example usage
if __name__ == "__main__":
    # Example usage of the GeminiVideoAnalyzer
    try:
        # Initialize the analyzer
        analyzer = GeminiVideoAnalyzer()
        
        # Example video path - replace with actual path
        video_path = "c:\\Users\\Zongjie\\Documents\\GitHub\\ugc\\test1.mp4" # Ensure this file exists
        
        # Check if the file exists before proceeding
        if os.path.exists(video_path):
            # Analyze the video
            analysis = analyzer.analyze_video(video_path)
            
            print("\n=== VIDEO ANALYSIS RESULT ===\n")
            print(f"Summary: {analysis.summary}\n")
            
            print("Hashtags:")
            for hashtag in analysis.hashtags:
                print(f"#{hashtag.tag} - {hashtag.category}")
            
            # You can also use the new get_hashtags method
            # hashtags = analyzer.get_hashtags(video_path)
            # print("\n=== HASHTAGS ONLY ===\n")
            # for hashtag in hashtags:
            #     print(f"#{hashtag.tag} - {hashtag.category}")
            
            # Example of using the style analyzer
            # style_json_path = "c:\\Users\\Zongjie\\Documents\\GitHub\\ugc\\upload\\style_analysis.json"
            # if os.path.exists(style_json_path):
            #     styled_analysis = analyzer.analyze_video_with_style(video_path, style_json_path)
            #     print("\n=== STYLED ANALYSIS RESULT ===\n")
            #     print(f"Styled Title: {styled_analysis.title}\n")
            #     print(f"Styled Summary: {styled_analysis.summary}\n")
            
        else:
            print(f"Video file not found: {video_path}")
            print("Please provide a valid video file path to analyze.")
    
    except Exception as e:
        print(f"Error: {str(e)}")