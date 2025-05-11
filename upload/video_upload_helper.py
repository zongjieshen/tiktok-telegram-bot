import os
import time
import logging
import asyncio
import aiohttp
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes, ConversationHandler, CallbackQueryHandler, MessageHandler, filters, CommandHandler

from upload.gemini_video_analyzer import GeminiVideoAnalyzer, Hashtag
from upload.upload_post_client import UploadPostClient, UploadPostError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Conversation states
# Add this to your conversation states
ANALYZING, REVIEWING, EDITING_TITLE, UPLOADING, WAITING_FOR_URL = range(5)

# Add this new function to handle the URL response
async def handle_transfer_url(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle the Pilk URL provided by the user."""
    chat_id = update.effective_chat.id
    
    # Check if we're expecting a URL from this user
    if chat_id not in active_uploads or active_uploads[chat_id]["status"] != "waiting_for_url":
        await update.message.reply_text("❌ I wasn't expecting a URL from you. Please use /uploadVideo to start the process.")
        return ConversationHandler.END
    
    # Get the URL from the message
    url = update.message.text.strip()
    
    # Basic validation of the URL
    if not url.startswith("http"):
        await update.message.reply_text(
            "❌ That doesn't look like a valid URL. Please send me the download link you received from Pilk."
        )
        return WAITING_FOR_URL
    
    # Create temp directory if it doesn't exist
    temp_dir = Path("temp_videos")
    temp_dir.mkdir(exist_ok=True)
    
    # Generate a unique filename
    video_filename = f"video_{chat_id}_{int(time.time())}.mp4"
    video_path = temp_dir / video_filename
    
    # Inform user that download is starting
    await update.message.reply_text("📥 Downloading your video...\nThis may take a few minutes.")
    
    try:
        # Download the video using aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise Exception(f"Failed to download video: HTTP {response.status}")
                
                # Save the video to disk
                with open(video_path, 'wb') as f:
                    while True:
                        chunk = await response.content.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
        
        # Inform user that analysis is starting
        await update.message.reply_text("🔍 Analyzing your video with AI...\nThis may take a few minutes.")
        
        # Store video info in context
        active_uploads[chat_id].update({
            "video_path": str(video_path),
            "status": "analyzing",
        })
        
        # Run the video analysis in a separate thread to avoid blocking
        analyzer = GeminiVideoAnalyzer()
        
        # Pass the local video path to the analyzer
        analysis = await asyncio.to_thread(analyzer.analyze_video, str(video_path))
        
        # Store analysis results
        active_uploads[chat_id].update({
            "status": "analyzed",
            "summary": analysis.summary,
            "hashtags": analysis.hashtags,
            "title": analysis.title
        })
        
        # Show results and ask for confirmation
        await show_analysis_results(update, context, chat_id)
        return REVIEWING
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
        await update.message.reply_text(f"❌ Error processing video: {str(e)}")
        
        # Clean up any temporary files if they were created
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception as del_e:
                logger.error(f"Error deleting temporary file: {str(del_e)}")
            
        return ConversationHandler.END

# Callback data
APPROVE = "approve_hashtags"
REMOVE_PREFIX = "remove_hashtag_"
EDIT_TITLE = "edit_title"
UPLOAD_VIDEO = "upload_video"
CANCEL = "cancel_upload"
BACK_TO_REVIEW = "back_to_review"  # Add this line

# Store for active uploads
active_uploads = {}

async def start_video_upload(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Start the video upload process by prompting user to send a video."""
    chat_id = update.effective_chat.id
    logger.info(f"Starting video upload process for chat_id: {chat_id}")
    
    # Define the Pilk URL as an environment variable
    TRANSFER_SH_URL = os.environ.get("TRANSFER_SH_URL", "http://192.168.20.223:8080/#")
    
    # Create keyboard with Pilk link
    keyboard = [
        [InlineKeyboardButton("📤 Upload via Pilk", url=TRANSFER_SH_URL)]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # Initialize the upload data
    active_uploads[chat_id] = {
        "status": "waiting_for_url",
        "platforms": ["tiktok"],  # Default platform
    }
    
    # Prompt user to use Pilk
    await update.message.reply_text(
        "📤 To upload your video, please follow these steps:\n\n"
        "1. Click the button below to open Pilk\n"
        "2. Upload your video using their interface\n"
        "3. Copy the download URL you receive\n"
        "4. Paste the URL back here\n\n"
        "You can cancel this process anytime by typing /cancel.",
        reply_markup=reply_markup
    )
    
    return WAITING_FOR_URL

async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancel the current conversation."""
    chat_id = update.effective_chat.id
    
    # Clean up any resources
    if chat_id in active_uploads:
        if "video_path" in active_uploads[chat_id]:
            try:
                os.remove(active_uploads[chat_id]["video_path"])
            except Exception as e:
                logger.error(f"Error deleting temporary file: {str(e)}")
        
        del active_uploads[chat_id]
    
    await update.message.reply_text("❌ Video upload process canceled.")
    return ConversationHandler.END


async def show_analysis_results(update: Update, context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> None:
    """Show analysis results and hashtag selection options."""
    upload_data = active_uploads[chat_id]
    
    # Create message with summary and title
    message = (
        f"✅ Video analysis complete!\n\n"
        f"📝 <b>Suggested Title:</b>\n{upload_data['title']}\n\n"
        f"🏷️ <b>Suggested Hashtags:</b>\n"
    )
    
    # Add hashtags
    for i, hashtag in enumerate(upload_data['hashtags']):
        message += f"{i+1}. {hashtag.tag}"
        message += "\n"
    
    # Create inline keyboard for hashtag selection
    keyboard = []
    
    # Add button to edit title
    keyboard.append([
        InlineKeyboardButton("✏️ Edit Title", callback_data=EDIT_TITLE)
    ])
    
    # Add buttons to remove individual hashtags
    for i, hashtag in enumerate(upload_data['hashtags']):
        keyboard.append([
            InlineKeyboardButton(f"❌ Remove {hashtag.tag}", callback_data=f"{REMOVE_PREFIX}{i}")
        ])
    
    # Add approve and cancel buttons
    keyboard.append([
        InlineKeyboardButton("✅ Approve All", callback_data=APPROVE),
        InlineKeyboardButton("❌ Cancel", callback_data=CANCEL)
    ])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # Check if this is a callback query or a regular message
    if update.callback_query:
        # For callback queries, edit the existing message
        await update.callback_query.edit_message_text(
            message,
            reply_markup=reply_markup,
            parse_mode='HTML'
        )
    else:
        # For regular messages, send a new message
        await update.message.reply_text(
            message,
            reply_markup=reply_markup,
            parse_mode='HTML'
        )

async def handle_hashtag_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle callback queries for hashtag selection."""
    query = update.callback_query
    await query.answer()
    
    chat_id = update.effective_chat.id
    callback_data = query.data
    
    if callback_data == CANCEL:
        # Cancel the upload process
        await query.edit_message_text("❌ Upload canceled.")
        
        # Clean up
        if chat_id in active_uploads:
            try:
                os.remove(active_uploads[chat_id]["video_path"])
            except Exception as e:
                logger.error(f"Error deleting temporary file: {str(e)}")
            
            del active_uploads[chat_id]
        
        return ConversationHandler.END
    
    elif callback_data == APPROVE:
        # User approved all hashtags, proceed to upload confirmation
        upload_data = active_uploads[chat_id]
        
        # Create confirmation message
        message = (
            f"🚀 Ready to upload your video!\n\n"
            f"📝 *Title:* {upload_data['title']}\n\n"
            f"🏷️ *Hashtags:* "
        )
        
        # Add hashtags
        hashtags_text = " ".join([f"{h.tag}" for h in upload_data['hashtags']])
        message += hashtags_text + "\n\n"
        
        # Add platform info
        platforms_text = ", ".join(upload_data['platforms'])
        message += f"📱 *Platforms:* {platforms_text}\n\n"
        
        # Create confirmation keyboard
        keyboard = [
            [
                InlineKeyboardButton("✅ Upload Now", callback_data=UPLOAD_VIDEO),
                InlineKeyboardButton("⬅️ Back", callback_data="back_to_review"),
                InlineKeyboardButton("❌ Cancel", callback_data=CANCEL)
            ]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Update message with confirmation
        await query.edit_message_text(
            message,
            reply_markup=reply_markup,
            parse_mode='HTML'
        )
        
        return UPLOADING
    
    elif callback_data == EDIT_TITLE:
        # Prompt user to enter a new title
        await query.edit_message_text(
            f"Please enter a new title for your video. Current title is:\n\n"
            f"<pre>{active_uploads[chat_id]['title']}</pre>\n\n"
            f"Type your new title below:",
            parse_mode='HTML'
        )
        return EDITING_TITLE
    
    elif callback_data.startswith(REMOVE_PREFIX):
        # Remove a specific hashtag
        index = int(callback_data[len(REMOVE_PREFIX):])
        upload_data = active_uploads[chat_id]
        
        # Remove the hashtag
        removed_hashtag = upload_data['hashtags'][index]
        upload_data['hashtags'] = [h for i, h in enumerate(upload_data['hashtags']) if i != index]
        
        # Update the message
        await query.edit_message_text(
            f"Removed #{removed_hashtag.tag}. Please wait...",
        )
        
        # Show updated results
        await show_analysis_results(update, context, chat_id)
        
        return REVIEWING

async def handle_upload_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle upload confirmation callback."""
    query = update.callback_query
    await query.answer()
    
    chat_id = update.effective_chat.id
    callback_data = query.data
    
    if callback_data == BACK_TO_REVIEW:
        # Show the previous review screen
        await show_analysis_results(update, context, chat_id)
        return REVIEWING
        
    if callback_data == CANCEL:
        # Cancel the upload process
        await query.edit_message_text("❌ Upload canceled.")
        
        # Clean up
        if chat_id in active_uploads:
            try:
                os.remove(active_uploads[chat_id]["video_path"])
            except Exception as e:
                logger.error(f"Error deleting temporary file: {str(e)}")
            
            del active_uploads[chat_id]
        
        return ConversationHandler.END
    
    elif callback_data == UPLOAD_VIDEO:
        # User confirmed upload, proceed with upload
        upload_data = active_uploads[chat_id]
        
        # Update status
        upload_data["status"] = "uploading"
        await query.edit_message_text("🚀 Uploading your video to social media platforms...")
        
        try:
            # Extract hashtag strings
            hashtags = [h.tag for h in upload_data['hashtags']]
            
            # Create description from summary
            description = upload_data['summary']
            
            # Initialize the upload client
            client = UploadPostClient()
            
            # Upload the video
            result = await asyncio.to_thread(
                client.upload_video,
                video_path=upload_data['video_path'],
                title=upload_data['title'],
                description=description,
                platforms=upload_data['platforms'],
                tags=hashtags
            )
            
            # Update status based on result
            if result and isinstance(result, dict) and result.get('success', False):
                # Upload was successful
                upload_data["status"] = "completed"
                upload_data["result"] = result
                
                # Send success message
                success_message = (
                    f"✅ Video uploaded successfully!\n\n"
                    f"📝 <b>Title:</b> {upload_data['title']}\n\n"
                    f"🏷️ <b>Hashtags:</b> {' '.join([h for h in hashtags])}\n\n"
                    f"📱 <b>Platforms:</b> {', '.join(upload_data['platforms'])}\n\n"
                )
                
                # Add any additional information from the result
                if result.get('post_urls'):
                    success_message += f"🔗 <b>Post URLs:</b>\n"
                    for platform, url in result['post_urls'].items():
                        success_message += f"- {platform}: {url}\n"
                
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=success_message,
                    parse_mode='HTML'
                )
            else:
                # Upload failed or returned unexpected result
                upload_data["status"] = "failed"
                upload_data["error"] = result.get('error', 'Unknown error occurred') if isinstance(result, dict) else 'Invalid response from upload service'
                
                # Send failure message
                error_message = (
                    f"❌ Video upload failed!\n\n"
                    f"<b>Error:</b> {upload_data['error']}\n\n"
                    f"Please try again later or contact support if the issue persists."
                )
                
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=error_message,
                    parse_mode='HTML'
                )
            
            # Clean up
            try:
                os.remove(upload_data['video_path'])
            except Exception as e:
                logger.error(f"Error deleting temporary file: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error uploading video: {str(e)}", exc_info=True)
            
            # Send error message
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"❌ Error uploading video: {str(e)}"
            )
            
            # Update status
            upload_data["status"] = "failed"
            upload_data["error"] = str(e)
        
        return ConversationHandler.END

async def handle_title_edit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle the new title input from the user."""
    chat_id = update.effective_chat.id
    new_title = update.message.text
    
    # Update the title in the active uploads
    active_uploads[chat_id]['title'] = new_title
    
    # Inform the user that the title has been updated
    await update.message.reply_text(
        f"✅ Title updated to:\n\n"
        f"*{new_title}*\n\n"
        f"Please wait while I update the information...",
        parse_mode='HTML'
    )
    
    # Show updated results
    await show_analysis_results(update, context, chat_id)
    
    return REVIEWING

def get_video_upload_handlers():
    """Return the handlers needed for video upload workflow."""
    return [
        ConversationHandler(
            entry_points=[CommandHandler("uploadVideo", start_video_upload)],
            states={
                WAITING_FOR_URL: [
                    MessageHandler(filters.TEXT & ~filters.COMMAND, handle_transfer_url),
                    CommandHandler("cancel", cancel_command)
                ],
                REVIEWING: [CallbackQueryHandler(handle_hashtag_callback)],
                EDITING_TITLE: [
                    MessageHandler(filters.TEXT & ~filters.COMMAND, handle_title_edit),
                    CommandHandler("cancel", cancel_command)
                ],
                UPLOADING: [CallbackQueryHandler(handle_upload_callback)],
            },
            fallbacks=[CommandHandler("cancel", cancel_command)],
            name="video_upload",
            persistent=False,
        )
    ]