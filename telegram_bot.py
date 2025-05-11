import os
import logging
import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters, ConversationHandler
from dotenv import load_dotenv
import json
from datetime import datetime
import hashlib
import telegramify_markdown

# Import the process_trending_ads function from your existing script
from trendingAds.tiktok_top_ads_scraper import process_trending_ads

# Import the video upload helper
from upload.video_upload_helper import get_video_upload_handlers

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Get the Telegram token from environment variables
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_TOKEN environment variable not set")

# Store for active tasks
active_tasks = {}

# Global cache for DataFrames
df_cache = {}

def generate_cache_key(params):
    """Generate a unique cache key based on the parameters."""
    # Create a string representation of the parameters
    param_str = json.dumps(params, sort_keys=True)
    # Create a hash of the parameters
    return hashlib.md5(param_str.encode()).hexdigest()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    await update.message.reply_text(
        "👋 Welcome to the TikTok Trending Ads Bot!\n\n"
        "Use /trendingAds to get trending TikTok ads.\n"
        "Use /uploadVideo to upload a video to social media platforms.\n"
        "Use /help to see all available commands."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    help_text = (
        "🤖 *TikTok Trending Ads Bot Commands*\n\n"
        "/trendingAds \- Get trending TikTok ads\. Use default parameters or provide custom ones\.\n"
        "Format \(optional\): `/trendingAds period region industry\_name limit order\_by keyword`\n"
        "Example \(default\): `/trendingAds`\n"
        "Example \(custom\): `/trendingAds 30 AU Tech 5 ctr fitness\_app`\n\n"
        "/uploadVideo \- Upload a video to social media platforms\n"
        "Format: Attach a video to this command\n"
        "Example: Send `/uploadVideo` with a video attachment\n\n"
        "/status \- Check the status of your last request\n"
        "/help \- Show this help message"
    )
    await update.message.reply_text(help_text, parse_mode='MarkdownV2')

async def run_trending_ads_task(update: Update, params: dict) -> None:
    """Run the trending ads task in the background and send updates."""
    chat_id = update.effective_chat.id
    task_id = f"task_{datetime.now().strftime('%Y%m%d%H%M%S')}_{chat_id}"

    # Generate a cache key for these parameters
    cache_key = generate_cache_key(params)
    
    # Store task info
    active_tasks[chat_id] = {
        "task_id": task_id,
        "status": "running",
        "params": params,
        "started_at": datetime.now().isoformat(),
        "command_type": "trending",  # Add command type identifier
        "cache_key": cache_key  # Store the cache key with the task
    }

    # Check if we have cached results for these parameters
    if cache_key in df_cache:
        logger.info(f"Using cached analysis for parameters: {params}")
        cached_data = df_cache[cache_key]
        df_with_analysis = cached_data['df']
        result_text = cached_data['summary']
        
        # Update task status
        active_tasks[chat_id]["status"] = "completed"
        active_tasks[chat_id]["completed_at"] = datetime.now().isoformat()
        active_tasks[chat_id]["df_with_analysis"] = df_with_analysis
        active_tasks[chat_id]["result"] = result_text
        
        await update.message.reply_text("✅ Retrieved analysis from cache!")
        
        # Format and send the result
        if isinstance(result_text, dict):
            result_text = json.dumps(result_text, indent=2)
        elif isinstance(result_text, str):
            result_text = result_text
        else:
            result_text = str(result_text)
            
        active_tasks[chat_id]["result"] = result_text
        
        formatted_result = telegramify_markdown.markdownify(result_text)
        await update.message.reply_text(f"✅ Analysis complete\!\n\n{formatted_result}", parse_mode='MarkdownV2')
        return

    await update.message.reply_text(f"🔍 Starting to fetch trending ads with parameters: {json.dumps(params, indent=2)}\n\nThis may take a few minutes...")

    try:
        # Run the process_trending_ads function with keyword support
        summary, df_with_analysis = await asyncio.to_thread(process_trending_ads,
                                       period=params.get('period'),
                                       region=params.get('region'),
                                       keyword=params.get('keyword'),
                                       output_file="tiktok_top_ads.csv",
                                       **{k:v for k,v in params.items() if k not in ['period', 'region', 'keyword']})

        # Update task status
        active_tasks[chat_id]["status"] = "completed"
        active_tasks[chat_id]["completed_at"] = datetime.now().isoformat()

        if summary and df_with_analysis is not None:
            # Format the result for Telegram message
            if isinstance(summary, dict):
                result_text = json.dumps(summary, indent=2)
            elif isinstance(summary, str):
                result_text = summary
            else:
                result_text = str(summary)

            # Store the result text and DataFrame in the active task
            active_tasks[chat_id]["result"] = result_text
            active_tasks[chat_id]["df_with_analysis"] = df_with_analysis
            
            # Export DataFrame to CSV and send it to user
            try:
                csv_path = f"analysis_results_{chat_id}.csv"
                df_with_analysis.to_csv(csv_path, index=False, encoding='utf-8-sig')
                
                # Send CSV file as document
                with open(csv_path, 'rb') as csv_file:
                    await update.message.reply_document(
                        document=csv_file,
                        filename="analysis_results.csv",
                        caption="📊 Analysis results in CSV format"
                    )
                
                # Clean up the temporary CSV file
                try:
                    os.remove(csv_path)
                    logger.info(f"Successfully removed CSV file: {csv_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove CSV file {csv_path}: {str(e)}")
                    
            except Exception as e:
                logger.error(f"Failed to send CSV file: {str(e)}")
                await update.message.reply_text("❌ Failed to generate CSV file from analysis results")
            
            # Store both summary and DataFrame in the global cache
            df_cache[cache_key] = {
                'summary': result_text,
                'df': df_with_analysis
            }
            logger.info(f"Stored analysis and summary in cache with key: {cache_key}")

                    
            # Upload videos if available in the DataFrame
            if 'video_path' in df_with_analysis.columns:
                video_paths = df_with_analysis['video_path'].dropna().tolist()
                if video_paths:
                    await update.message.reply_text("📹 Uploading video examples in parallel...")
                    
                    # Define an async function for uploading a single video
                    async def upload_single_video(video_path, index):
                        if os.path.exists(video_path):
                            try:
                                # For smaller files, send as video with increased timeouts
                                with open(video_path, 'rb') as video_file:
                                    await update.message.reply_video(
                                        video=video_file,
                                        caption=f"Example video {index+1} from analysis",
                                        supports_streaming=True,
                                        read_timeout=240,
                                        write_timeout=240,
                                        connect_timeout=30,
                                        pool_timeout=120
                                    )
                                
                                logger.info(f"Successfully uploaded video: {video_path}")
                                # Remove the video file after successful upload
                                try:
                                    os.remove(video_path)
                                    logger.info(f"Successfully removed video file: {video_path}")
                                except Exception as e:
                                    logger.warning(f"Failed to remove video file {video_path}: {str(e)}")
                                return True
                            except Exception as e:
                                logger.error(f"Failed to upload video {video_path}: {str(e)}")
                                await update.message.reply_text(f"❌ Failed to upload video {index+1}: {str(e)}")
                                
                                # Try to send a link to the local file as fallback
                                try:
                                    file_path = os.path.abspath(video_path)
                                    await update.message.reply_text(
                                        f"📁 Video file is available locally at: `{file_path}`",
                                        parse_mode="MarkdownV2"
                                    )
                                except Exception:
                                    pass
                                return False
                        else:
                            logger.warning(f"Video file not found: {video_path}")
                            return False
                    
                    # Create tasks for all videos
                    upload_tasks = [upload_single_video(path, i) for i, path in enumerate(video_paths)]
                    
                    # Run all uploads in parallel with a limit of 3 concurrent uploads
                    # to avoid overwhelming Telegram's API
                    results = []
                    for i in range(0, len(upload_tasks), 3):
                        batch = upload_tasks[i:i+3]
                        batch_results = await asyncio.gather(*batch, return_exceptions=True)
                        results.extend(batch_results)
                    
                    # Count successful uploads
                    successful = sum(1 for r in results if r is True)
                    await update.message.reply_text(f"✅ Uploaded {successful} of {len(video_paths)} videos")
                else:
                    logger.info("No video paths found in the DataFrame")
                    
            # Format the text for Telegram MarkdownV2
            formatted_result = telegramify_markdown.markdownify(result_text)
            await update.message.reply_text(f"✅ Analysis complete\!\n\n{formatted_result}", parse_mode='MarkdownV2')

        else:
            await update.message.reply_text("❌ No results returned. There might have been an error or no ads found.")
            active_tasks[chat_id]["status"] = "failed"
            active_tasks[chat_id]["error"] = "No results returned"

    except Exception as e:
        logger.error(f"Error in trending ads task: {str(e)}", exc_info=True)
        await update.message.reply_text(f"❌ Error: {str(e)}")
        active_tasks[chat_id]["status"] = "failed"
        active_tasks[chat_id]["error"] = str(e)
        active_tasks[chat_id]["completed_at"] = datetime.now().isoformat()

async def trending_ads(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Get trending TikTok ads with default or custom parameters."""
    args = context.args
    
    # Initialize with default parameters
    params = {
        "period": 30,
        "region": "AU",
        "order_by": "ctr",
        "limit": 5,
        "keyword": None
    }
    
    # If arguments are provided, update parameters
    if args:
        # Update parameters based on provided arguments
        if len(args) >= 1 and args[0].isdigit():
            params["period"] = int(args[0])
        
        if len(args) >= 2:
            params["region"] = args[1]
        
        if len(args) >= 3:
            params["industry_name"] = args[2]
        
        if len(args) >= 4 and args[3].isdigit():
            params["limit"] = int(args[3])
        
        if len(args) >= 5:
            params["order_by"] = args[4]
        
        if len(args) >= 6:
            params["keyword"] = args[5]
    
    await run_trending_ads_task(update, params)

async def check_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Check the status of the last request."""
    chat_id = update.effective_chat.id
    
    if chat_id in active_tasks:
        task = active_tasks[chat_id]
        status = task["status"]
        
        if status == "running":
            started_at = datetime.fromisoformat(task["started_at"])
            elapsed = (datetime.now() - started_at).total_seconds()
            await update.message.reply_text(
                f"🕒 Task is still running (elapsed: {elapsed:.1f} seconds)\n"
                f"Parameters: {json.dumps(task['params'], indent=2)}"
            )
        elif status == "completed":
            await update.message.reply_text("✅ Your last task was completed successfully.")
        elif status == "failed":
            await update.message.reply_text(
                f"❌ Your last task failed with error: {task.get('error', 'Unknown error')}"
            )
    else:
        await update.message.reply_text("No active or recent tasks found for you.")



def main() -> None:
    """Start the bot."""
    # Create the Application
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Add command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("trendingAds", trending_ads))
    application.add_handler(CommandHandler("status", check_status))
    
    # Add video upload handlers
    for handler in get_video_upload_handlers():
        application.add_handler(handler)

    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
