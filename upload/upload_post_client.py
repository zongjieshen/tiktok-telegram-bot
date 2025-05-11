import os
import logging
import requests
from pathlib import Path
from typing import List, Dict, Union, Optional
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

class UploadPostError(Exception):
    """Base exception for Upload-Post API errors"""
    pass

class UploadPostClient:
    """
    A client for the Upload-Post API to upload videos to social media platforms.
    """
    
    BASE_URL = "https://api.upload-post.com/api"
    
    def __init__(self, api_key=None):
        """
        Initialize the UploadPostClient with API key.
        
        Args:
            api_key (str, optional): Upload-Post API key. If not provided, will try to get from environment variable.
        """
        # Get API key from parameter or environment variable
        self.api_key = api_key or os.getenv("UPLOAD_POST_API_KEY")
        
        if not self.api_key:
            logger.error("No Upload-Post API key provided. Please set UPLOAD_POST_API_KEY environment variable or pass it to the constructor.")
            raise ValueError("Upload-Post API key is required")
        
        # Initialize session with authorization header
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Apikey {self.api_key}",
            "User-Agent": "upload-post-python-client/1.0.0"
        })
        
        logger.info("UploadPostClient initialized successfully")
    
    def upload_video(
        self,
        video_path: Union[str, Path],
        title: str,
        description: Optional[str] = None,
        user: str = None,
        platforms: List[str] = ["tiktok"],
        tags: Optional[List[str]] = None
    ) -> Dict:
        """
        Upload a video to specified social media platforms
        
        Args:
            video_path (str or Path): Path to video file
            title (str): Video title
            description (str, optional): Video description
            user (str, optional): User identifier, defaults to value from environment variable
            platforms (List[str]): List of platforms (e.g. ["tiktok", "instagram"])
            tags (List[str], optional): List of hashtags
            
        Returns:
            Dict: API response JSON
            
        Raises:
            UploadPostError: If upload fails
        """
        video_path = Path(video_path)
        if not video_path.exists():
            error_msg = f"Video file not found: {video_path}"
            logger.error(error_msg)
            raise UploadPostError(error_msg)
        
        # Get user from parameter or environment variable
        user = user or os.getenv("UPLOAD_POST_USER")
        
        logger.info(f"Uploading video: {video_path} to platforms: {platforms}")
        
        try:
            # Prepare multipart form data
            files = {"video": open(video_path, "rb")}
            
            # Modify title to include hashtags if provided
            modified_title = title
            if tags and len(tags) > 0:
                hashtag_string = " " + " ".join([f"#{tag}" if not tag.startswith('#') else tag for tag in tags])
                modified_title = title + hashtag_string
                logger.info(f"Added hashtags to title: {modified_title}")
            
            # Prepare form data
            data = {
                "title": modified_title,
                "user": user,
            }
            
            # Add optional parameters if provided
            if description:
                data["description"] = description
            
            # Add platforms as platform[] parameters
            for platform in platforms:
                data[f"platform[]"] = platform
            
            # Make the POST request
            logger.info(f"Sending request to {self.BASE_URL}/upload")
            response = self.session.post(
                f"{self.BASE_URL}/upload",
                files=files,
                data=data
            )
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # Parse and return JSON response
            result = response.json()
            logger.info(f"Upload successful. Response: {result}")
            return result
            
        except requests.exceptions.RequestException as e:
            error_msg = f"API request failed: {str(e)}"
            logger.error(error_msg)
            raise UploadPostError(error_msg)
        except (ValueError, TypeError) as e:
            error_msg = f"Invalid response format: {str(e)}"
            logger.error(error_msg)
            raise UploadPostError(error_msg)
        finally:
            # Ensure file is closed
            if 'files' in locals() and 'video' in files:
                files['video'].close()

