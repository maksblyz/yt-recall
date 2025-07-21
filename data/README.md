# Data Folder

This folder contains user-generated recall data and analysis results.

## Structure

- `recalls/` - Contains saved recall responses and analysis in JSON format
  - Files are named as `{video_id}_{timestamp}.json`
  - Each file contains the complete recall session data

## File Format

Each recall file contains:
- `timestamp` - When the recall was performed
- `video_url` - Original YouTube URL
- `video_title` - Video title
- `video_id` - YouTube video ID
- `user_recall` - User's recall text
- `transcript` - Full video transcript
- `comparison` - AI analysis results
- `score` - Numerical score (0-100)
- `transcript_length` - Length of transcript in characters
- `recall_length` - Length of user recall in characters

## Privacy

- This folder is gitignored to prevent accidental commits of user data
- All data is stored locally on your machine
- No data is sent to external servers (except for transcription and analysis APIs)

## Usage

Users can save their recall results using the "Save Results" button in the app. This creates a timestamped JSON file in the `recalls/` folder that can be reviewed later or used for tracking learning progress. 