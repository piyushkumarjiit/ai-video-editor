# YouTube Video Uploader Setup

## Overview
Upload scale model car building videos to your YouTube channel (@modernhackers) using the YouTube Data API v3.

## Initial Setup

### 1. Install Required Packages
```bash
pip install google-api-python-client google-auth-oauthlib google-auth-httplib2
```

### 2. Get YouTube API Credentials

1. **Go to Google Cloud Console**: https://console.cloud.google.com/

2. **Create or Select Project**:
   - Click "Select a project" → "New Project"
   - Name it (e.g., "YouTube Uploader")
   - Click "Create"

3. **Enable YouTube Data API v3**:
   - Go to "APIs & Services" → "Library"
   - Search for "YouTube Data API v3"
   - Click on it and press "Enable"

4. **Configure OAuth Consent Screen** (if not done yet):
   - Go to "APIs & Services" → "OAuth consent screen"
   - User Type: **External** (required for non-Workspace accounts)
   - Click "Create"
   - **Page 1 - App information**:
     - App name: "modernhackers" (or your app name)
     - User support email: your Google account email
     - Developer contact: your Google account email
     - Click "SAVE AND CONTINUE"
   - **Page 2 - Scopes**:
     - Click "ADD OR REMOVE SCOPES"
     - Search for "youtube" and select: `https://www.googleapis.com/auth/youtube.upload`
     - Click "UPDATE" → "SAVE AND CONTINUE"
   - **Page 3 - Test users** (IMPORTANT):
     - Click "+ ADD USERS"
     - Enter your Google account email (the one that owns your YouTube channel)
     - Click "ADD" → "SAVE AND CONTINUE"
     - **Note**: Without adding yourself as a test user, you'll get "Access blocked" errors
   - **Page 4 - Summary**:
     - Review and click "BACK TO DASHBOARD"

5. **Create OAuth 2.0 Credentials**:
   - Go to "APIs & Services" → "Credentials"
   - Click "Create Credentials" → "OAuth client ID"
   - Application type: **Desktop app**
   - Name: "YouTube Upload Client"
   - Click "Create"

6. **Download Credentials**:
   - Click the download icon (⬇️) next to your OAuth 2.0 Client ID
   - Save the file as `client_secrets.json` in `/home/mazsola/video/`

### 3. Verify Your YouTube Channel

**IMPORTANT**: The Google account you authenticate with MUST:
- Have a YouTube channel created and configured
- Have accepted YouTube's Terms of Service
- Be able to upload videos (not restricted)

To verify:
1. Go to https://www.youtube.com/ and log in
2. Click your profile icon → "Your channel"
3. If you see "Create a channel", click it and set up your channel
4. Upload a test video manually to confirm your account works

### 4. First Authentication
```bash
cd /home/mazsola/video
python upload_youtube.py --video /path/to/test_video.mp4 --title "Test Video"
```

This will:
- Open a browser window
- Ask you to log in to your Google account
- Request permissions to upload videos
- Save credentials to `youtube_credentials.json` for future use

## Usage

### Basic Upload (Private)
```bash
python upload_youtube.py --video /home/mazsola/Videos/timeline_output_4k_youtube.mp4
```

### Upload with Custom Metadata
```bash
python upload_youtube.py \
  --video /home/mazsola/Videos/timeline_output_4k_youtube.mp4 \
  --title "Build Alpha Model Giulia GTAm - Candy Red (Part 3)" \
  --description "Scale model build with detailed painting and finishing techniques" \
  --tags "scale model,car modeling,alfa romeo,painting,airbrush" \
  --privacy public
```

### Upload to Specific Playlist
```bash
python upload_youtube.py \
  --video /home/mazsola/Videos/timeline_output_4k_youtube.mp4 \
  --playlist PLxxxxxxxxxxxxxxxxx \
  --privacy unlisted
```

## Configuration

Edit `project_config.json` to set defaults:

```json
{
  "youtube": {
    "channel_url": "https://www.youtube.com/@modernhackers",
    "default_title": "Scale Model Car Build",
    "default_description": "Building scale model cars with detailed painting and finishing techniques.\n\nYou can follow ModernHackers social media @ https://linktr.ee/modernhackers",
    "default_tags": ["scale model", "car model", "car modeling", "painting", "airbrush"],
    "category_id": "26",
    "default_privacy": "private",
    "default_playlist_id": null
  }
}
```

### Category IDs
- `2` = Autos & Vehicles
- `22` = People & Blogs
- `24` = Entertainment
- `26` = Howto & Style (default - for scale modeling tutorials)
- `28` = Science & Technology

### Privacy Options
- `private` - Only you can see
- `unlisted` - Anyone with link can see
- `public` - Everyone can see

## Command-Line Options

```
--video VIDEO           Path to video file (required)
--title TITLE          Video title
--description TEXT     Video description
--tags TAGS            Comma-separated tags
--privacy STATUS       public/private/unlisted
--category ID          YouTube category ID
--playlist ID          Add to playlist ID
--config FILE          Config file (default: project_config.json)
--client-secrets FILE  OAuth secrets (default: client_secrets.json)
--credentials FILE     Stored credentials (default: youtube_credentials.json)
```

## Integration with Render Pipeline

After rendering completes, upload automatically:

```bash
# Render video
python render_youtube.py --output /home/mazsola/Videos/output.mp4

# Upload to YouTube
python upload_youtube.py \
  --video /home/mazsola/Videos/output.mp4 \
  --title "RC Car Build - $(date +%Y-%m-%d)" \
  --privacy unlisted
```

## Troubleshooting

### "Access blocked: [app name] has not completed the Google verification process"
**Cause**: You haven't added yourself as a test user in the OAuth consent screen.

**Solution**:
1. Go to: https://console.cloud.google.com/apis/credentials/consent
2. In left sidebar, click "OAuth consent screen"
3. Look for "Audience" section or click the "Audience" tab
4. Scroll to "Test users" section
5. Click "+ ADD USERS"
6. Enter your Google account email
7. Click "ADD" or "SAVE"
8. Delete `youtube_credentials.json` and re-run the upload command

### "youtubeSignupRequired" or HTTP 401 Unauthorized
**Cause**: The Google account doesn't have a YouTube channel or can't upload videos.

**Solution**:
1. Go to https://www.youtube.com/ with the Google account you're using
2. Check if you have a channel created (click profile icon → "Your channel")
3. If no channel exists, create one
4. Try uploading a video manually through YouTube to verify your account works
5. Delete `youtube_credentials.json` and authenticate again with the correct account

### Wrong Google Account Authenticated
**Solution**:
```bash
rm youtube_credentials.json
python upload_youtube.py --video /path/to/video.mp4 --title "Test"
# Browser will open - log in with the CORRECT Google account that owns your YouTube channel
```

### "Client secrets file not found"
- Download OAuth credentials from Google Cloud Console
- Save as `client_secrets.json` in video directory

### "The request cannot be completed because you have exceeded your quota"
- YouTube API has daily upload quota (default: 10,000 units)
- One video upload ≈ 1,600 units
- Request quota increase in Google Cloud Console if needed

### "Access Not Configured"
- Make sure YouTube Data API v3 is enabled in Google Cloud Console

### Browser doesn't open for OAuth
- Check firewall settings
- Try running on local machine instead of remote server
- Use `--noauth_local_webserver` flag (requires manual code entry)

### Set Chrome as Default Browser (Linux)
If Firefox opens but you want Chrome:
```bash
# For Flatpak Chrome
xdg-settings set default-web-browser com.google.Chrome.desktop

# For system Chrome
xdg-settings set default-web-browser google-chrome.desktop
```

## Important Notes

### OAuth App Status
- Your app will remain in "Testing" status (this is normal and preferred)
- You can have up to 100 test users
- App verification is NOT required if you stay in Testing mode
- Only test users can authenticate with your app

### Account Requirements
- The Google account MUST have a YouTube channel
- The channel must be able to upload videos (not restricted/suspended)
- You can verify by uploading a video manually through YouTube first

### Security Notes
- `client_secrets.json` - Contains OAuth client ID/secret (keep private)
- `youtube_credentials.json` - Contains access token (keep private, auto-refreshes)
- Add both to `.gitignore`
- Never commit these files to version control

## Files Created

- `upload_youtube.py` - Main upload script
- `client_secrets.json` - OAuth credentials (you provide)
- `youtube_credentials.json` - Stored access token (auto-generated)
