# 📄 BART Article Summarizer

A powerful text summarization application using a fine-tuned BART model with history tracking, chunked processing, and ROUGE score evaluation.

🌐 Live Demo
Try it now: https://huggingface.co/spaces/Sanarin1334/cnnmail-summary
Click the link above to use the application directly in your browser - no installation required!

## 🌟 Features

- **Advanced Text Summarization**: Uses fine-tuned BART model (`Sanarin1334/cnnmail-model`)
- **Chunked Processing**: Handles long articles by splitting into manageable chunks
- **Chaining Strategy**: Combines chunk summaries for optimal final output
- **ROUGE Evaluation**: Provides ROUGE-1, ROUGE-2, and ROUGE-L scores
- **History Management**: Automatic saving and display of recent summaries
- **Word Count Validation**: Ensures minimum 50-word input requirement
- **Interactive Web Interface**: Built with Gradio for easy use
- **Copy Functionality**: Copy button for easy summary sharing
- **Example Articles**: Pre-loaded sample texts for testing

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Step 1: Clone or Download

Download the `app.py` file to your local machine.

### Step 2: Install Dependencies

```bash
pip install transformers torch gradio evaluate rouge-score
```

### Step 3: Run the Application

```bash
python app.py
```

The application will start and provide a local URL (typically `http://127.0.0.1:7860`).

## 📋 Usage Guide

### Basic Usage

1. **Start the Application**
   ```bash
   python app.py
   ```

2. **Open Your Browser**
   - Navigate to the provided local URL
   - The interface will load automatically

3. **Enter Text**
   - Paste your article in the input textbox
   - Minimum 50 words required
   - Word counter shows progress

4. **Generate Summary**
   - Click "🔍 Generate Summary" button
   - Wait for processing (may take 30-60 seconds)
   - View the generated summary and ROUGE scores

### Advanced Features

#### Example Articles
- Use pre-loaded sample articles for testing
- Click any example button to load content
- Topics include: Climate Change, AI, Space Exploration

#### History Management
- **View History**: Recent summaries are automatically displayed
- **Refresh History**: Click "🔄 Refresh" to update display
- **Clear History**: Click "🗑️ Clear History" to remove all records
- **Automatic Saving**: Last 50 summaries are saved automatically

#### ROUGE Scores
The application provides three ROUGE metrics:
- **ROUGE-1**: Word-level overlap between summary and original
- **ROUGE-2**: Bigram-level overlap assessment
- **ROUGE-L**: Longest common subsequence evaluation

## 🛠️ Technical Details

### Model Information
- **Base Model**: BART (Bidirectional and Auto-Regressive Transformers)
- **Fine-tuned Model**: `Sanarin1334/cnnmail-model`
- **Optimized For**: News articles and formal text

### Processing Pipeline
1. **Input Validation**: Check minimum word count
2. **Text Chunking**: Split long texts into 400-word chunks
3. **Individual Summarization**: Generate summary for each chunk
4. **Summary Chaining**: Combine and re-summarize for final output
5. **Quality Assessment**: Calculate ROUGE scores
6. **History Recording**: Save results automatically

### File Structure
```
├── app.py                 # Main application file
├── summary_history.json   # Auto-generated history file (created on first use)
└── README.md             # This file
```

## ⚙️ Configuration

### Model Parameters
- **Max Input Length**: 1024 tokens per chunk
- **Chunk Size**: 400 words maximum
- **Beam Search**: 6 beams for better quality
- **Length Penalty**: 1.0 (balanced length control)
- **Repetition Penalty**: 1.1 (reduces repetition)

### Customization Options
You can modify these parameters in `app.py`:

```python
# Chunk size adjustment
max_words = 400  # Change chunk size

# Summary length control
min_length = 100        # Minimum summary length
max_new_tokens = 400    # Maximum summary length

# Generation parameters
num_beams = 6           # Beam search width
length_penalty = 1.0    # Length control
repetition_penalty = 1.1 # Repetition reduction
```

## 🔧 Troubleshooting

### Common Issues

**Model Loading Error**
```
Solution: Ensure stable internet connection for model download
First run may take 5-10 minutes to download model files
```

**Memory Issues**
```
Solution: Reduce chunk size or max_new_tokens
Consider using a machine with more RAM (8GB+ recommended)
```

**Slow Processing**
```
Solution: Use GPU if available (CUDA-compatible)
Reduce num_beams parameter for faster generation
```

**History File Errors**
```
Solution: Check write permissions in application directory
Delete summary_history.json and restart application
```

## 💡 Tips for Best Results

### Input Guidelines
- **Optimal Length**: 200-2000 words work best
- **Text Type**: News articles, reports, academic papers
- **Language**: English text only
- **Format**: Plain text without special formatting

### Quality Optimization
- Use complete sentences and paragraphs
- Avoid heavily technical jargon
- Ensure clear topic focus
- Remove unnecessary formatting

## 🎯 Performance Metrics

### Expected Processing Times
- **Short Articles** (50-200 words): 10-20 seconds
- **Medium Articles** (200-800 words): 30-45 seconds
- **Long Articles** (800+ words): 60-90 seconds

### ROUGE Score Interpretation
- **ROUGE-1 > 0.3**: Good word overlap
- **ROUGE-2 > 0.15**: Good phrase preservation
- **ROUGE-L > 0.25**: Good structural similarity

## 📄 License

This project uses the Hugging Face Transformers library and the specified pre-trained model. Please check the individual model and library licenses for usage terms.

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify all dependencies are installed correctly
3. Ensure Python version compatibility
4. Check internet connection for model downloads

## 🔄 Updates and Improvements

The application includes:
- Automatic history management (last 50 summaries)
- Real-time word counting
- Error handling and user feedback
- Responsive web interface
- Copy functionality for easy sharing

---

**Ready to summarize?** Run `python app.py` and start creating concise, high-quality summaries!
