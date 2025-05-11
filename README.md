# TikTok Telegram Bot

A Telegram bot for analyzing trending TikTok ads, uploading videos to social media platforms. This project leverages AI and various APIs to automate content analysis for UGC (User Generated Content) creators and marketers.

---

## Features

### Telegram Bot Commands

- **/start**  
  Welcomes the user and provides a quick overview of available commands.

- **/help**  
  Displays detailed help for all bot commands, including usage examples.

- **/trendingAds**  
  Fetches trending TikTok ads using default or custom parameters.  
  - **Usage:**  
    `/trendingAds`  
    `/trendingAds period region industry_name limit order_by keyword`  
  - **Example:**  
    `/trendingAds 30 AU Tech 5 ctr fitness_app`

- **/uploadVideo**  
  Uploads a video to social media platforms.  
  - **Usage:**  
    Send `/uploadVideo` with a video attachment.

---

## Core Functionalities

### Trending Ads Analysis
You will need to create an account on RapidApi and subscribe the end point of Tiktok Creative Center API https://rapidapi.com/Lundehund/api/tiktok-creative-center-api/playground/apiendpoint_395c8f73-b5d8-4ebb-8bca-257dec46cf4c. There's a free tier. 

#### Analysis Features
- Performs deep analysis using Google's Gemini API to extract:
  - Structural breakdown (shots, scenes, hooks)
  - Visual editing style (colors, transitions, text overlays)
  - Audio patterns (music, voice-overs, sound effects)
  - Content structure (narrative flow, keywords)
  - Engagement tactics (questions, social proof, CTAs)

#### Results Processing
- Generates comprehensive CSV reports
- Provides video examples with analysis
- Implements caching for faster repeated queries
- Supports concurrent processing of multiple videos

### Content Generation & Style Analysis
Leverages Google Gemini's advanced AI capabilities to analyze and generate UGC-optimized content, with automated caption and hashtag generation.

#### Style Analysis and Caption Writing
- AI-powered content generation following extracted styles
- Style-consistent content transformation
- Maintains factual accuracy while adapting tone
- Structured output with:
  - Tone and pacing guidelines
  - Vocabulary recommendations
  - Punctuation and emoji patterns
  - Hook and CTA templates
- Brand voice preservation across content
- Adapts content while preserving key messaging
- Provides structured style recommendations

### Video Upload
This uses the UPLOAD POST API to upload videos to Instagram and TikTok.

#### Features
- Basic upload functionality
- Progress tracking
- Status monitoring
- Error handling


## Requirements

- Python 3.10+
- Docker (for containerized deployment)
- Telegram Bot Token
- API keys for:
  - RapidAPI (Instagram/TikTok scraping)
  - Google GenAI (Gemini)
  - UPLOAD POST

See `requirements.txt` for all Python dependencies.

---

## Deployment Guide

### 1. Clone the Repository

```bash
git clone https://github.com/zongjieshen/tiktok-telegram-bot.git
cd tiktokTelegram

create a .env file in the project root with the following variables or load those variables into docker-compose.yml
TELEGRAM_TOKEN=your_telegram_bot_token
RAPIDAPI_KEY=your_rapidapi_key
GOOGLE_API_KEY=your_google_genai_key
UPLOAD_POST_API_KEY=your_upload_post_api_key
UPLOAD_POST_USER=your_upload_post_user
TIKTOK_ACCOUNT=your_tiktok_account
EMAIL=your_email
INSTAGRAM_ACCOUNT=your_instagram_account

### 3. Start the Application
docker-compose up -d

