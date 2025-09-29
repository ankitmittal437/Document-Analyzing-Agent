# Document Chat System

A comprehensive document ingestion and chat interface application that allows users to upload various types of documents (PDF, Word, images, URLs) and interact with them through a conversational AI interface.

## Features

- **Multi-format Document Support**: Upload PDFs, Word documents, text files, and images
- **URL Processing**: Extract and process content from web URLs
- **Vector Database Storage**: Uses ChromaDB for efficient document storage and retrieval
- **AI-Powered Chat Interface**: Ask questions about your documents using OpenAI's GPT models
- **Conversation Memory**: Maintains context across chat sessions
- **Source Attribution**: Shows which documents were used to answer your questions
- **Real-time Statistics**: Track document count and query statistics

## Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Tesseract OCR (for image text extraction)

## Installation

1. **Clone or download this repository**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Tesseract OCR**:
   
   **Windows**:
   - Download from: https://github.com/UB-Mannheim/tesseract/wiki
   - Add to PATH or set `TESSERACT_CMD` environment variable
   
   **macOS**:
   ```bash
   brew install tesseract
   ```
   
   **Linux**:
   ```bash
   sudo apt-get install tesseract-ocr
   ```

4. **Set up OpenAI API Key**:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
   
   Or create a `.env` file:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

## Usage

### Option 1: Unified Application (Recommended)
Run the complete system with both document upload and chat interface:
```bash
streamlit run app.py
```

### Option 2: Separate Applications
Run document ingestion and chat interface separately:

**Document Ingestion**:
```bash
streamlit run document_ingestion.py
```

**Chat Interface**:
```bash
streamlit run chat_interface.py
```

## How to Use

### 1. Upload Documents
- Navigate to the "Document Upload" page
- Upload files using the file uploader (supports PDF, DOCX, TXT, PNG, JPG, JPEG, GIF, BMP)
- Or enter a URL to process web content
- The system will automatically extract text and store it in the vector database

### 2. Chat with Documents
- Switch to the "Chat Interface" page
- Ask questions about your uploaded documents
- The AI will search through your documents and provide answers with source attribution
- View the sources used for each answer in the expandable sections

### 3. Manage Your Data
- View collection statistics in the sidebar
- Clear chat history when needed
- Adjust the number of documents to retrieve for better results

## Supported File Types

- **PDF**: Text extraction from PDF documents
- **Word Documents**: .docx files
- **Text Files**: Plain text files (.txt)
- **Images**: PNG, JPG, JPEG, GIF, BMP (with OCR text extraction)
- **URLs**: Web pages and online documents

## Configuration

### Adjusting Document Processing
Edit the `RecursiveCharacterTextSplitter` parameters in the code:
- `chunk_size`: Size of text chunks (default: 1000)
- `chunk_overlap`: Overlap between chunks (default: 200)

### Customizing AI Model
Modify the ChatOpenAI configuration:
- `model_name`: Choose between "gpt-3.5-turbo" or "gpt-4"
- `temperature`: Adjust creativity (0.0 to 1.0)
- `max_tokens`: Maximum response length

## Troubleshooting

### Common Issues

1. **"No module named 'pytesseract'"**
   - Install pytesseract: `pip install pytesseract`
   - Ensure Tesseract OCR is installed and accessible

2. **"OpenAI API key not found"**
   - Set your OpenAI API key as an environment variable
   - Or create a `.env` file with your key

3. **"Error reading PDF"**
   - Ensure PDF is not password-protected
   - Try with a different PDF file

4. **"Error reading image"**
   - Ensure Tesseract OCR is properly installed
   - Check if the image contains readable text

### Performance Tips

- Use smaller chunk sizes for better precision
- Increase chunk overlap for better context
- Limit the number of retrieved documents for faster responses
- Clear chat history periodically to free up memory

## File Structure

```
document_chat_app/
├── app.py                    # Main unified application
├── document_ingestion.py     # Document upload interface
├── chat_interface.py         # Chat interface
├── requirements.txt          # Python dependencies
├── README.md                # This file
└── chroma_db/               # Vector database storage (created automatically)
```

## Dependencies

- `streamlit`: Web interface framework
- `langchain`: LLM application framework
- `chromadb`: Vector database
- `openai`: OpenAI API client
- `PyPDF2`: PDF text extraction
- `python-docx`: Word document processing
- `beautifulsoup4`: HTML parsing for URLs
- `pytesseract`: OCR for image text extraction
- `pillow`: Image processing

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source and available under the MIT License.
