import os
import requests
import pandas as pd
import logging # <-- Add this

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # <-- Add this
logger = logging.getLogger(__name__) # <-- Add this

class TikTokRapidAPIClient:
    def __init__(self):
        self.base_url = "https://tiktok-creative-center-api.p.rapidapi.com/api/trending/ads"
        self.api_key = os.getenv("RAPIDAPI_KEY")  # Make sure to set RAPIDAPI_KEY in your .env file
        self.headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": "tiktok-creative-center-api.p.rapidapi.com"
        }
        if not self.api_key:
            logger.error("RAPIDAPI_KEY environment variable not set.") # <-- Add this check

    def get_trending_ads(self, **params):
        """
        Fetch trending TikTok ads from RapidAPI.
        Args:
            **params: Query parameters for the API.
        Returns:
            list: List of ad data dictionaries formatted for DataFrame conversion, or None if failed.
        """
        if not self.api_key: # <-- Add this check
            logger.error("Cannot fetch trending ads: RAPIDAPI_KEY is not configured.")
            return None

        try:
            logger.info(f"Fetching trending ads with params: {params}") # <-- Add this
            response = requests.get(self.base_url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Check if the response has the expected structure
            if data.get("code") != 0 or "data" not in data or "materials" not in data["data"]:
                logger.error(f"Unexpected API response structure: {data}") # <-- Change print to logger.error
                return None

            # Get current date for timestamp
            current_date = pd.Timestamp.now().strftime('%Y-%m-%d')

            # Process each material to extract relevant information
            processed_data = []
            materials = data["data"].get("materials", []) # <-- Safer access
            logger.info(f"Received {len(materials)} materials from API.") # <-- Add this
            for material in materials:
                # Extract basic ad information
                ad_info = {
                    "date": current_date,
                    "id": material.get("id"),
                    "ad_title": material.get("ad_title"),
                    "brand_name": material.get("brand_name"),
                    "cost": material.get("cost"),
                    "ctr": material.get("ctr"),
                    "like": material.get("like"),
                    "industry_key": material.get("industry_key"),
                    "objective_key": material.get("objective_key")
                }
                
                # Extract video information if available
                if "video_info" in material:
                    video_info = material["video_info"]
                    ad_info["vid"] = video_info.get("vid")
                    ad_info["duration"] = video_info.get("duration")
                    ad_info["cover"] = video_info.get("cover")
                    ad_info["width"] = video_info.get("width")
                    ad_info["height"] = video_info.get("height")
                    
                    # Select the best available video URL resolution
                    # Try 720p first, then fall back to lower resolutions
                    video_url = None
                    if "video_url" in video_info:
                        resolutions = ["720p", "540p", "480p", "360p"]
                        for res in resolutions:
                            if res in video_info["video_url"] and video_info["video_url"][res]:
                                video_url = video_info["video_url"][res]
                                ad_info["resolution"] = res
                                break
                    
                    ad_info["video_url"] = video_url
                
                processed_data.append(ad_info)

            logger.info(f"Successfully processed {len(processed_data)} ads.") # <-- Add this
            return processed_data

        except requests.exceptions.RequestException as e: # <-- More specific exception
            logger.error(f"Error fetching trending ads from RapidAPI: {e}") # <-- Change print to logger.error
            return None
        except Exception as e: # <-- Catch other potential errors
            logger.error(f"An unexpected error occurred in get_trending_ads: {e}", exc_info=True) # <-- Log traceback
            return None