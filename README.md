# YouTube Video Q&A with RAG

An intelligent Streamlit application that enables users to ask questions about any YouTube video content using Retrieval-Augmented Generation (RAG) powered by Embedchain and GPT-4o.

## Overview

This application transforms YouTube videos into an interactive Q&A experience. By leveraging vector embeddings and large language models, users can extract insights, summaries, and specific information from video content without watching the entire video.

## Features

- 🎥 **YouTube Video Processing**: Extract and analyze content from any YouTube video URL
- 💬 **Interactive Q&A**: Ask natural language questions about video content
- 🤖 **RAG-Powered Answers**: Get accurate, context-aware responses using Retrieval-Augmented Generation
- 🔄 **Vector Embeddings**: Efficient storage and retrieval using Embedchain
- ⚡ **Fast Response Time**: Quick answers without re-watching videos
- 🎯 **Customizable LLM**: Support for multiple language models (GPT-4o, GPT-3.5, and more)

## Technology Stack

- **Streamlit**: Web application framework
- **Embedchain**: Vector database and embedding management
- **OpenAI GPT-4o**: Large Language Model for generating responses
- **Python**: Core programming language
- **YouTube Transcript API**: Video content extraction

## How It Works

1. **Video Input**: User provides a YouTube video URL
2. **Content Extraction**: The app extracts transcript/content from the video
3. **Embedding Creation**: Content is converted into vector embeddings using Embedchain
4. **Storage**: Embeddings are stored in a vector database for efficient retrieval
5. **Question Processing**: User asks questions about the video
6. **RAG Pipeline**: 
   - Retrieves relevant video segments based on the question
   - Augments the query with retrieved context
   - Generates accurate answers using GPT-4o
7. **Response**: User receives contextual answers based on video content

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key

### Setup

```bash
# Clone the repository

# Navigate to project directory
cd youtube-video-qa-rag

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### Requirements.txt

```
streamlit
embedchain
openai
youtube-transcript-api
python-dotenv
```

## Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

### Running the Application

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

### Using the App

1. **Enter YouTube URL**: Paste the URL of the YouTube video you want to analyze
2. **Wait for Processing**: The app will extract and process the video content
3. **Ask Questions**: Type your questions in the chat interface
4. **Get Answers**: Receive accurate, context-aware responses based on video content

### Example Questions

- "What are the main topics covered in this video?"
- "Can you summarize the key points?"
- "What did the speaker say about [specific topic]?"
- "What examples were provided for [concept]?"
- "What are the conclusions or takeaways?"

## Project Structure

```
youtube-video-qa-rag/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Project dependencies
├── .env                    # Environment variables (not in repo)
├── .gitignore             # Git ignore file
├── README.md              # Project documentation
│
├── utils/
│   ├── __init__.py
│   ├── video_processor.py  # YouTube video processing
│   └── rag_engine.py       # RAG implementation
│
└── data/
    └── embeddings/         # Stored vector embeddings (gitignored)
```

## Code Example

```python
import streamlit as st
from embedchain import App

# Initialize Embedchain app
@st.cache_resource
def get_ec_app():
    return App.from_config(config={
        "llm": {
            "provider": "openai",
            "config": {
                "model": "gpt-4o",
                "temperature": 0.5,
            }
        }
    })

# Streamlit interface
st.title("🎥 YouTube Video Q&A Assistant")

# Input YouTube URL
video_url = st.text_input("Enter YouTube Video URL:")

if video_url:
    app = get_ec_app()
    
    # Add video to knowledge base
    with st.spinner("Processing video..."):
        app.add(video_url, data_type="youtube_video")
    
    st.success("Video processed! Ask your questions below.")
    
    # Q&A interface
    question = st.text_input("Ask a question about the video:")
    
    if question:
        with st.spinner("Generating answer..."):
            response = app.query(question)
        
        st.write("**Answer:**")
        st.write(response)
```

## Customization

### Changing the LLM

You can easily switch to different language models:

```python
# Use GPT-3.5 Turbo
config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-3.5-turbo",
            "temperature": 0.5,
        }
    }
}

# Use other providers (Anthropic Claude, Cohere, etc.)
config = {
    "llm": {
        "provider": "anthropic",
        "config": {
            "model": "claude-3-opus-20240229",
        }
    }
}
```

### Adjusting Response Parameters

```python
config = {
    "llm": {
        "config": {
            "temperature": 0.7,      # Creativity (0-1)
            "max_tokens": 500,       # Response length
            "top_p": 0.9,           # Nucleus sampling
        }
    }
}
```

## Features in Detail

### RAG (Retrieval-Augmented Generation)

RAG combines the power of retrieval systems with generative AI:

1. **Retrieval**: Finds relevant video segments matching the question
2. **Augmentation**: Provides context to the language model
3. **Generation**: Creates accurate, grounded answers

This approach ensures responses are:
- Factually accurate (based on actual video content)
- Contextually relevant
- Free from hallucinations

### Vector Embeddings

Embedchain automatically:
- Converts video content into numerical representations
- Stores embeddings efficiently
- Enables semantic search across video content
- Supports similarity-based retrieval

## Limitations

- Requires videos with available transcripts/captions
- Processing time depends on video length
- API costs associated with OpenAI usage
- May not capture visual-only content (charts, demonstrations)

## Future Enhancements

- [ ] Support for multiple videos in a single session
- [ ] Video timestamp references in answers
- [ ] Multi-language support for international videos
- [ ] Export Q&A sessions as PDF reports
- [ ] Integration with other video platforms (Vimeo, Dailymotion)
- [ ] Visual content analysis using multimodal models
- [ ] Conversation history and context retention
- [ ] Batch processing for multiple videos

