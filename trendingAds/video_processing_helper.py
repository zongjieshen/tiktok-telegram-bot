import requests
import os
import time
import json
import pandas as pd
import logging
import concurrent.futures
from functools import partial
from google import genai
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define Pydantic models for structured output
class ShotScene(BaseModel):
    timestamp: str
    description: str

class Hook(BaseModel):
    description: str
    attention_grabber: str

class CallToAction(BaseModel):
    timing: str
    style: str
    description: str

class StructuralBreakdown(BaseModel):
    shot_scene_segmentation: List[ShotScene]
    hook_detection: Hook
    call_to_action: Optional[CallToAction] = None

class VisualStyle(BaseModel):
    color_profile: str
    transition_types: List[str]
    text_overlay_patterns: str

class AudioPattern(BaseModel):
    music_style: str
    voice_over_style: str
    sound_effects: Optional[str] = None

class ContentSegment(BaseModel):
    segment_type: str
    duration: str

class ContentStructure(BaseModel):
    topic_sequence: str
    keywords: List[str]
    key_segments: List[ContentSegment]

class EngagementTactic(BaseModel):
    question_prompts: Optional[List[str]] = None
    social_proof_elements: Optional[List[str]] = None
    urgency_scarcity_triggers: Optional[List[str]] = None

class VideoStructureAnalysis(BaseModel):
    structural_breakdown: StructuralBreakdown
    visual_editing_style: VisualStyle
    audio_patterns: AudioPattern
    content_structure: ContentStructure
    engagement_tactics: EngagementTactic

def download_video(url, output_path):
    """Download video from URL to a local file."""
    try:
        #logger.info(f"Attempting to download video from {url} to {output_path}") # <-- Add logging
        response = requests.get(url, stream=True, timeout=60) # <-- Add timeout
        response.raise_for_status()

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"Video downloaded successfully to {output_path}") # <-- Change print to logger.info
        return True
    except requests.exceptions.RequestException as e: # <-- More specific exception
        logger.error(f"Error downloading video from {url}: {str(e)}") # <-- Change print to logger.error
        return False
    except Exception as e:
        logger.error(f"Unexpected error downloading video {url}: {str(e)}", exc_info=True) # <-- Log other errors
        return False

def analyze_video(video_path):
    """Analyse video using Google Gemini API."""
    try:
        logger.info(f"Analyzing video: {video_path}")
        
        # Initialize Gemini client
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        
        # Upload the video file
        logger.info(f"Uploading video file: {video_path}...")
        display_name = os.path.basename(video_path)
        video_file = client.files.upload(
            file=video_path,
            config={
                "display_name": display_name,
                "mime_type": "video/mp4"  # Assuming mp4, adjust if needed
            }
        )
        logger.info(f"Uploaded file '{video_file.display_name}' as: {video_file.name}")
        
        # Wait for the file to be processed and become ACTIVE
        logger.info("Waiting for file to become ACTIVE...")
        while video_file.state.name == "PROCESSING":
            logger.info("File is still PROCESSING. Waiting 5 seconds...")
            time.sleep(5)  # Wait for 5 seconds
            video_file = client.files.get(name=video_file.name)  # Fetch the latest file state
            logger.info(f"Current state: {video_file.state.name}")
        
        if video_file.state.name == "ACTIVE":
            logger.info(f"File '{video_file.name}' is now ACTIVE and ready for use.")
            
            # Create the prompt parts
            prompt_parts = [
                """Analyze this TikTok/social media video and provide a detailed breakdown of its structure and patterns.
                
                Please provide analysis in the following categories:
                
                1. Structural Breakdown:
                   - Shot/Scene Segmentation: Identify major cuts and transitions with timestamps
                   - Hook Detection: Describe the first 3 seconds - what grabs attention (text, action, question)
                   - Call-to-Action (CTA): Identify any CTAs, their timing, and style
                
                2. Visual & Editing Style:
                   - Color Profile: Describe dominant colors, filters, or visual effects
                   - Transition Types: List all transitions used (cuts, wipes, fades, etc.)
                   - Text Overlay Patterns: Describe text style, placement, fonts, emojis
                
                3. Audio Patterns:
                   - Music Style: Describe background music genre, tempo, mood
                   - Voice-Over Style: Describe narration pacing, tone, style
                   - Sound Effects: Note any distinctive sound effects and their timing
                
                4. Content Structure:
                   - Topic Sequence: Break down the narrative flow (intro → problem → solution → CTA)
                   - Keywords: Extract 5-7 key terms or phrases that define the content
                   - Duration of Key Segments: Note timing of introduction, main content, conclusion
                
                5. Engagement Tactics:
                   - Question Prompts: Identify direct questions to viewers
                   - Social Proof Elements: Note mentions of popularity, testimonials, etc.
                   - Urgency/Scarcity Triggers: Identify time-limited offers or exclusivity claims
                
                Format your response as structured data with clear sections and bullet points.
                """,
                video_file  # Pass the File object directly
            ]
            
            # Generate content with structured output
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt_parts,
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': VideoStructureAnalysis,
                }
            )
            
            # Get the structured analysis
            analysis = response.parsed
            
            logger.info(f"Analysis successful for {video_path}")
            return analysis
            
        elif video_file.state.name == "FAILED":
            error_message = f"File '{video_file.name}' processing FAILED."
            if hasattr(video_file, 'state_reason') and video_file.state_reason:
                error_message += f" Reason: {video_file.state_reason}"
            logger.error(error_message)
            return None
        else:
            error_message = f"File '{video_file.name}' is in an unexpected state: {video_file.state.name}"
            logger.error(error_message)
            return None
            
    except Exception as e:
        logger.error(f"Error analysing video {video_path}: {str(e)}", exc_info=True)
        return None

def process_single_video(row, i, total_videos, temp_dir):
    """Process a single video - download and analyze."""
    processed = False
    failed_reason = None
    analysis = None
    
    # Check if analysis already exists and is valid
    if not pd.isna(row.get('analysis')) and row['analysis'] not in ["Analysis failed"]:
        logger.info(f"Video {i+1}/{total_videos}: Skipping - analysis already exists")
        return {
            'index': i,
            'processed': False,
            'skipped_existing': True,
            'skipped_invalid': False,
            'failed_download': False,
            'failed_analysis': False,
            'analysis': row.get('analysis')
        }

    video_url = row.get('video_url')  # Use .get for safety

    # Skip if not a valid URL
    if pd.isna(video_url) or not isinstance(video_url, str) or not video_url.startswith("http"):
        logger.warning(f"Video {i+1}/{total_videos}: Skipping - invalid or missing video URL.")
        return {
            'index': i,
            'processed': False,
            'skipped_existing': False,
            'skipped_invalid': True,
            'failed_download': False,
            'failed_analysis': False,
            'analysis': "Invalid URL"
        }

    logger.info(f"Video {i+1}/{total_videos}: Processing {video_url}")

    # Download the video
    video_filename = f"video_{row.get('id', i)}.mp4"  # Use ad ID if available for filename
    video_path = os.path.join(temp_dir, video_filename)

    if download_video(video_url, video_path):
        # Analyze the video
        logger.info(f"Video {i+1}/{total_videos}: Attempting analysis...")
        analysis = analyze_video(video_path)

        if analysis:
            logger.info(f"Video {i+1}/{total_videos}: Analysis successful.")
            processed = True
        else:
            analysis = "Analysis failed"
            logger.warning(f"Video {i+1}/{total_videos}: Analysis failed or returned failure message.")
            failed_reason = "analysis"

        # Instead of removing the video file, store its path for reference
        logger.info(f"Video {i+1}/{total_videos}: Keeping video file for reference: {video_path}")
        
        # Return the video path along with other results
        return {
            'index': i,
            'processed': processed,
            'skipped_existing': False,
            'skipped_invalid': False,
            'failed_download': False,
            'failed_analysis': failed_reason == "analysis",
            'analysis': analysis,
            'video_path': video_path  # Add the video path to the result
        }
    else:
        logger.error(f"Video {i+1}/{total_videos}: Download failed: {video_url}")
        analysis = "Download failed"
        failed_reason = "download"
    
    logger.info(f"Video {i+1}/{total_videos}: Finished processing.")
    
    # Return a dictionary with all the results
    return {
        'index': i,
        'processed': processed,
        'skipped_existing': False,
        'skipped_invalid': False,
        'failed_download': failed_reason == "download",
        'failed_analysis': failed_reason == "analysis",
        'analysis': analysis
    }

def process_videos_for_analysis(df, max_workers=5, output_path=None):
    """Process all videos from DataFrame concurrently and analyze them."""
    temp_dir = "temp_videos"
    # Create temp directory if it doesn't exist
    if not os.path.exists(temp_dir):
        try:
            os.makedirs(temp_dir)
            logger.info(f"Created temporary directory: {temp_dir}")
        except OSError as e:
            logger.error(f"Failed to create temporary directory {temp_dir}: {e}")
            return df  # Return early if temp dir cannot be created
        
    # Add video_path column to store paths to downloaded videos
    if 'video_path' not in df.columns:
        df['video_path'] = None
    # Add analysis columns if they don't exist
    if 'analysis' not in df.columns:
        df['analysis'] = None
    
    # Add individual analysis component columns
    analysis_columns = [
        'structural_breakdown', 'visual_style', 'audio_patterns',
        'content_structure', 'engagement_tactics'
    ]
    for col in analysis_columns:
        if col not in df.columns:
            df[col] = None
    
    # Process each video
    total_videos = len(df)
    processed_count = 0
    skipped_existing = 0
    skipped_invalid = 0
    failed_download = 0
    failed_analysis = 0

    # Create a partial function with fixed parameters
    process_func = partial(process_single_video, total_videos=total_videos, temp_dir=temp_dir)
    
    # Process videos concurrently using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        tasks = [(row, i) for i, row in df.iterrows()]
        results = list(executor.map(lambda args: process_func(*args), tasks))
        
    # Process results and update DataFrame
    for result in results:
        i = result['index']
        
        # Update DataFrame with results
        analysis = result['analysis']
        if isinstance(analysis, VideoStructureAnalysis):
            # Store the full analysis object as JSON string
            df.at[i, 'analysis'] = analysis.json()
            
            # Store individual components for easier analysis
            df.at[i, 'structural_breakdown'] = analysis.structural_breakdown.json()
            df.at[i, 'visual_style'] = analysis.visual_editing_style.json()
            df.at[i, 'audio_patterns'] = analysis.audio_patterns.json()
            df.at[i, 'content_structure'] = analysis.content_structure.json()
            df.at[i, 'engagement_tactics'] = analysis.engagement_tactics.json()
            
            # Store the video path if available
            if 'video_path' in result:
                df.at[i, 'video_path'] = result['video_path']
        else:
            # Handle error cases
            df.at[i, 'analysis'] = str(analysis)
            for col in analysis_columns:
                df.at[i, col] = None
        
        # Update counters
        if result['processed']:
            processed_count += 1
        if result['skipped_existing']:
            skipped_existing += 1
        if result['skipped_invalid']:
            skipped_invalid += 1
        if result['failed_download']:
            failed_download += 1
        if result['failed_analysis']:
            failed_analysis += 1

    logger.info(f"Video processing summary: Total={total_videos}, Newly Processed={processed_count}, "
                f"Skipped (Existing)={skipped_existing}, Skipped (Invalid URL)={skipped_invalid}, "
                f"Failed Downloads={failed_download}, Failed Analyses={failed_analysis}")
    
    # Save to CSV with proper encoding
    try:
        if output_path is None:
            output_path = "video_analysis_results.csv"
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"Successfully saved analysis results to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save results to CSV: {str(e)}")
    
    return df

def generate_overall_summary(df):
    """Generate an overall summary of all video analyses."""
    
    # Generate summary based on valid analyses
    try:
        required_columns = [
            'structural_breakdown', 'visual_style', 'audio_patterns',
            'content_structure', 'engagement_tactics'
        ]
        if not all(col in df.columns for col in required_columns):
            return "Missing required analysis columns."

        # Filter out rows with no analysis or failed analysis
        valid_analyses = df[~df[required_columns].isna().all(axis=1)]
        if len(valid_analyses) == 0:
            return "No valid analysis data available for summary generation."
            
        # Calculate average video duration if duration column exists
        avg_duration = None
        if 'duration' in df.columns and not df['duration'].isna().all():
            # Convert duration to float and calculate average
            durations = df['duration'].dropna().astype(float)
            if len(durations) > 0:
                avg_duration = durations.mean()
                logger.info(f"Average video duration: {avg_duration:.2f} seconds")
        
        # Prepare the five fields as lists of JSON strings
        structural_breakdowns = valid_analyses['structural_breakdown'].tolist()
        visual_styles = valid_analyses['visual_style'].tolist()
        audio_patterns = valid_analyses['audio_patterns'].tolist()
        content_structures = valid_analyses['content_structure'].tolist()
        engagement_tactics = valid_analyses['engagement_tactics'].tolist()

        # Compose the prompt for Gemini
        prompt = f"""
        You are a professional video producer. I'm giving you five fields from our best-performing TikTok ads, each as a JSON list (each item is one ad):

        • structural_breakdown: {json.dumps(structural_breakdowns)}
        • visual_style: {json.dumps(visual_styles)}
        • audio_patterns: {json.dumps(audio_patterns)}
        • content_structure: {json.dumps(content_structures)}
        • engagement_tactics: {json.dumps(engagement_tactics)}
        """
        
        # Add average duration information if available
        if avg_duration is not None:
            prompt += f"""
        • average_duration: {avg_duration:.2f} seconds
            
        Important: The average video length is {avg_duration:.2f} seconds. Use this to create a realistic timeline.
            """
            
        prompt += """
        1. **Summary (1-2 sentences):**
        Condense these five attributes into a single cohesive creative concept.

        2. **Filming Guide:** For each section below, provide only 2-4 concise bullet points.
        - **Shots/Angles (with exact timeline based on the average duration, e.g., \"0-3s: intro shot, 3-7s: demo, ...\"):** (max 4 bullet points)
        - **Lighting:** (max 2 bullet points)
        - **Audio/VO:** (max 2 bullet points)
        - **Editing tips:** (max 3 bullet points)
        - **On-screen text & CTA:** (max 2 bullet points)

        Output in this format:

        Summary:
        <your summary here>

        Filming Guide:

        Shots/Angles:
        <your answer here>

        Lighting:
        <your answer here>

        Audio/VO:
        <your answer here>

        Editing Tips:
        <your answer here>

        On-screen Text & CTA:
        <your answer here>
        """

        # Call Gemini for the summary
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        summary = response.text
        
        # Create the final summary with video references
        final_summary = f"""# Video Filming Playbook

Analysis based on {len(valid_analyses)} videos

{summary}"""
        
        return final_summary

    except Exception as e:
        logger.error(f"Error generating overall summary: {str(e)}")
        return f"Summary generation failed: {str(e)}"

