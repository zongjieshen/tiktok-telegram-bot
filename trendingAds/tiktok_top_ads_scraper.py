import pandas as pd
import os
import json
from dotenv import load_dotenv
import logging

# Import the video processing functions from the new helper file
from trendingAds.video_processing_helper import process_videos_for_analysis, generate_overall_summary

# Import the TikTok RapidAPI client
from trendingAds.tiktok_rapidapi_client import TikTokRapidAPIClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # <-- Add this
logger = logging.getLogger(__name__) # <-- Add this

# Load environment variables (for API keys)
load_dotenv()

# Modified scrape_tiktok_top_ads function to use RapidAPI
# Define default parameters at module level for consistency
DEFAULT_PARAMS = {
    "period": 30,
    "region": "AU",
    "limit": 3,
    "page": 1,
    "order_by": "ctr",
    "industry_name": None
}

# Load industry mapping from JSON file
def load_industry_mapping():
    """
    Load industry mapping from the JSON file.

    Returns:
        dict: Dictionary mapping industry names to industry codes
    """
    try:
        json_path = os.path.join(os.path.dirname(__file__), "tiktok-ads-industry-code.json")
        with open(json_path, 'r') as f:
            industry_data = json.load(f)

        # Create mapping dictionary for main industries
        industry_mapping = {}
        for industry in industry_data:
            industry_mapping[industry["name"]] = industry["id"]

            # Add sub-industries to the mapping
            if "sub_industry" in industry:
                for sub_industry in industry["sub_industry"]:
                    industry_mapping[sub_industry["name"]] = sub_industry["id"]

        logger.info("Successfully loaded industry mapping.") # <-- Add logging
        return industry_mapping
    except FileNotFoundError:
        logger.error(f"Industry mapping file not found at {json_path}") # <-- Log specific error
        return {}
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from industry mapping file: {json_path}") # <-- Log specific error
        return {}
    except Exception as e:
        logger.error(f"Error loading industry mapping: {str(e)}", exc_info=True) # <-- Change print to logger.error and add traceback
        # Fallback to empty dictionary if file can't be loaded
        return {}

# Load the industry mapping
INDUSTRY_MAPPING = load_industry_mapping()

# Function to convert industry name to industry code
def get_industry_code(industry_name):
    """
    Convert an industry name to its corresponding industry code.

    Args:
        industry_name (str): The name of the industry
        
    Returns:
        str: The industry code, or None if not found
    """
    if not industry_name:
        return None
        
    # Check for exact match
    if industry_name in INDUSTRY_MAPPING:
        return INDUSTRY_MAPPING[industry_name]
    
    # Try to find closest match using string similarity
    best_match = None
    highest_similarity = 0
    
    for name, code in INDUSTRY_MAPPING.items():
        # Simple substring matching
        if industry_name.lower() in name.lower() or name.lower() in industry_name.lower():
            # Calculate a simple similarity score based on length of common substring
            name_lower = name.lower()
            industry_lower = industry_name.lower()
            
            # Check if one is substring of the other
            if name_lower in industry_lower:
                similarity = len(name_lower) / len(industry_lower)
            elif industry_lower in name_lower:
                similarity = len(industry_lower) / len(name_lower)
            else:
                # Calculate word overlap
                name_words = set(name_lower.split())
                industry_words = set(industry_lower.split())
                common_words = name_words.intersection(industry_words)
                
                if common_words:
                    similarity = len(common_words) / max(len(name_words), len(industry_words))
                else:
                    similarity = 0
            
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = code
    
    # Return the best match if found, otherwise None
    return best_match

# Modified scrape_tiktok_top_ads function to use default parameters
def scrape_tiktok_top_ads(period=DEFAULT_PARAMS["period"],
                         region=DEFAULT_PARAMS["region"],
                         limit=DEFAULT_PARAMS["limit"],
                         page=DEFAULT_PARAMS["page"],
                         order_by=DEFAULT_PARAMS["order_by"],
                         output_file=None,
                         **kwargs):
    """
    Get top performing Ads videos from TikTok Creative Center using RapidAPI.

    Args:
        period (int): Time period in days (7, 30, 180) (default: 30)
        region (str): Region/country code (AU, US, etc.) (default: "AU")
        keyword (str, optional): Keyword to search for
        limit (int): Number of results per page (default: 20)
        page (int): Page number for pagination (default: 1)
        order_by (str): Sorting criteria (ctr, impression, like, play_6s_rate, cvr, etc.) (default: "ctr")
        output_file (str, optional): Path to save the CSV output file
        **kwargs: Additional parameters to pass to the API:
            - like (str, optional): Filter by like range (1=Top 1-20%, 2=Top 21-40%, etc.)
            - ad_format (str, optional): Filter by ad format (1=spark ads, 2=non-spark ads)
            - objective (str, optional): Filter by objective (1=Traffic, 2=App installs, etc.)
            - industry_name (str, optional): Industry name to filter by
            - industry (str, optional): Industry code
            - ad_language (str, optional): Ad language code

    Returns:
        pd.DataFrame: DataFrame containing scraped data with analytics URLs, or None if failed.
    """
    try:
        logger.info("Initializing TikTok RapidAPI client...") # <-- Change print to logger.info
        # Initialize the TikTok RapidAPI client
        client = TikTokRapidAPIClient()

        # Set up parameters for the API call
        params = {
            "period": period,
            "country": region,
            "limit": limit,
            "page": page,
            "order_by": order_by,
            "ad_language":"en"
        }
        
        # Convert industry_name to industry code if provided
        if "industry_name" in kwargs and kwargs["industry_name"] is not None:
            industry_code = get_industry_code(kwargs["industry_name"])
            if industry_code:
                params["industry"] = industry_code
                logger.info(f"Converted industry name '{kwargs['industry_name']}' to code '{industry_code}'") # <-- Change print to logger.info
            else:
                logger.warning(f"Could not find industry code for name: {kwargs['industry_name']}") # <-- Add warning

        # List of optional parameters to check for
        optional_params = ["like", "ad_format", "objective", "industry", "ad_language", "keyword"]
        
        # Add optional parameters if provided
        for param in optional_params:
            if param in kwargs and kwargs[param] is not None:
                params[param] = kwargs[param]
        
        # Add any additional parameters from kwargs
        params.update(kwargs)
        
        # Use the trending_ads endpoint for general top ads
        logger.info(f"Fetching top ads for region {region} over {period} days...") # <-- Change print to logger.info
        ads_data = client.get_trending_ads(**params)

        if not ads_data:
            logger.warning("No valid ad data could be extracted from the API response.") # <-- Change print to logger.warning
            return None

        # Create DataFrame from the collected data
        df = pd.DataFrame(ads_data)
        logger.info(f"Step 1: Collected {len(df)} ads data.") # <-- Change print to logger.info

        # Check for duplicates in the DataFrame itself
        if 'id' in df.columns:
            initial_count = len(df)
            df = df.drop_duplicates(subset=['id'], keep='first')
            duplicates_removed = initial_count - len(df)
            if duplicates_removed > 0:
                logger.info(f"Removed {duplicates_removed} duplicate records based on 'id'.")
                
        # Write DataFrame to CSV if output_file is provided
        if output_file:
            # Write to file
            df.to_csv(output_file, index=False)
            logger.info(f"New data saved to {output_file}, replacing previous content")

        return df

    except Exception as e:
        logger.error(f"An unexpected error occurred during Step 1 (scrape_tiktok_top_ads): {str(e)}", exc_info=True) # <-- Change print to logger.error and add traceback
        return None


def process_trending_ads(period=DEFAULT_PARAMS["period"], 
                         region=DEFAULT_PARAMS["region"], 
                         keyword=None, 
                         output_file="tiktok_top_ads.csv", 
                         **scrape_params):
    """
    Process trending TikTok ads with scraping and analysis.
    
    Args:
        period (int): Time period in days (7, 30, 180) (default: 30)
        region (str): Region/country code (AU, US, etc.) (default: "AU")
        keyword (str, optional): Keyword to search for
        output_file (str): Path to save the CSV output file (default: "tiktok_top_ads.csv")
        **scrape_params: Additional parameters to pass to the scraper:
            - order_by (str): Sorting criteria (default: "ctr")
            - industry_name (str): Industry name to filter by (default: "Apps")
            - limit (int): Number of results per page
            - page (int): Page number for pagination
            - like (str, optional): Filter by like range
            - ad_format (str, optional): Filter by ad format
            - objective (str, optional): Filter by objective
            - industry (str, optional): Industry code
            - ad_language (str, optional): Ad language code
    
    Returns:
        tuple: (str, pd.DataFrame) - A tuple containing the summary of the analysis and the DataFrame with analysis data,
               or (None, None) if failed.
    """
    try:
        # Set default scrape parameters if not provided
        for param, value in DEFAULT_PARAMS.items():
            if param not in scrape_params and param != "region" and param != "period":
                scrape_params[param] = value
        
        # Add keyword to scrape_params if provided
        if keyword:
            scrape_params["keyword"] = keyword
        
        logger.info(f"Using configuration: period={period}, region={region}, keyword={keyword}, output={output_file}") # <-- Changed line

        # Step 1: Scrape TikTok Top Ads
        logger.info("--- Starting Step 1: Scrape TikTok Top Ads ---") # <-- Change print to logger.info
        logger.info(f"Parameters: period={period} days, region={region}, keyword={keyword}, other params: {scrape_params}") # Use logger for details

        df_with_videos = scrape_tiktok_top_ads(period=period, region=region, output_file=output_file, **scrape_params)

        if df_with_videos is None or df_with_videos.empty: # <-- Check if empty
            logger.error("Failed to scrape TikTok top ads or no ads found.") # <-- Change print to logger.error
            return None, None

        # Ensure we have the latest data, especially if appended
        try:
            df_read = pd.read_csv(output_file)
            logger.info(f"Read {len(df_read)} records from {output_file} for processing.")
        except FileNotFoundError:
             logger.error(f"Output file '{output_file}' not found after scraping.")
             return None, None
        except Exception as e:
             logger.error(f"Error reading output file '{output_file}': {e}", exc_info=True)
             return None, None


        # Step 3: Process videos for transcripts and analysis
        logger.info("\n--- Starting Step 3: Generate Transcripts and Analysis ---")

        # Process videos for transcripts and analysis with concurrent processing
        # Pass max_workers parameter to enable concurrent processing (adjust value as needed)
        df_with_analysis = process_videos_for_analysis(df_read, max_workers=5, output_path=output_file)

        # Save the final DataFrame with transcripts and analysis
        summary = generate_overall_summary(df_with_analysis)
        if output_file:
            try:
                df_with_analysis.to_csv(output_file, index=False)
                logger.info(f"--- Step 3 finished: Final data saved to {output_file} ---") # <-- Change print to logger.info
            except Exception as e:
                 logger.error(f"Failed to save final data to {output_file}: {e}", exc_info=True)

        logger.info("\nProcessing completed successfully.") # <-- Change print to logger.info
        return summary, df_with_analysis

    except FileNotFoundError as e:
        logger.error(f"Error: Could not find required file: {str(e)}", exc_info=True) # <-- Change print to logger.error and add traceback
        return None, None
    except Exception as e:
        logger.error(f"An unexpected error occurred in process_trending_ads: {str(e)}", exc_info=True) # <-- Change print to logger.error and add traceback
        return None, None


if __name__ == "__main__":
    # Hardcoded default values for command-line execution
    period = 7
    region = "AU"
    keyword = None
    scrape_params = {
        "order_by": "ctr",
        "industry_name": "Apps"
    }
    final_output_file = "tiktok_top_ads.csv"
    
    # Call the function with the hardcoded parameters
    result = process_trending_ads(
        period=period,
        region=region,
        keyword=keyword,
        output_file=final_output_file,
        **scrape_params
    )
    
    if result is None:
        print("\nScript finished with errors.")
        exit(1)
    else:
        print("\nScript finished successfully.")