# app.py *-* Full BART Summarizer with History, Chaining, and ROUGE Scores (English Version)

from transformers import BartTokenizer, BartForConditionalGeneration
import json
import os
from datetime import datetime
import gradio as gr
import evaluate

# ----------------------
# MODEL SETUP
# ----------------------
model_name = "Sanarin1334/cnnmail-model"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# ROUGE metric
rouge_metric = evaluate.load("rouge")

# ----------------------
# HISTORY MANAGEMENT
# ----------------------
HISTORY_FILE = "summary_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []

def save_history(history):
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history[-50:], f, indent=2, ensure_ascii=False)
    except:
        pass

def clear_history():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    return True

def get_history():
    history = load_history()
    if not history:
        return "No previous summaries."
    
    history_text = ""
    for i, entry in enumerate(reversed(history[-10:])):
        history_text += f"**{i+1}. {entry['timestamp']}**\n"
        history_text += f"Input ({entry['input_words']} words): {entry['input']}\n"
        history_text += f"Summary ({entry['summary_words']} words): {entry['summary'][:150]}...\n"
        history_text += "---\n\n"
    return history_text

# ----------------------
# HELPER FUNCTIONS
# ----------------------
def count_words(text):
    return len(text.split()) if text else 0

def split_into_chunks(text, max_words=400):
    words = text.split()
    return [' '.join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

def generate_summary(chunk, min_length=100, max_new_tokens=400):
    inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=1024, padding=True)
    summary_ids = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=max_new_tokens,
        min_length=min_length,
        num_beams=6,
        do_sample=False,
        length_penalty=1.0,
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()

# ----------------------
# SUMMARIZE FUNCTION
# ----------------------
def summarize_text(text):
    if not text.strip():
        return {'error': 'Please enter text to summarize.', 'summary': '', 'word_count': 0}

    word_count = count_words(text)
    if word_count < 50:
        return {'error': f'Please enter at least 50 words. Current: {word_count} words.', 'summary': '', 'word_count': word_count}
    
    chunks = split_into_chunks(text, max_words=400)
    summaries = []

    try:
        # Summarize each chunk
        for chunk in chunks:
            summaries.append(generate_summary(chunk, min_length=80, max_new_tokens=300))
        
        # Combine all chunk summaries
        combined_summary = " ".join(summaries)

        # Chaining: summarize the combined summary
        final_summary = generate_summary(combined_summary, min_length=100, max_new_tokens=350)

        summary_word_count = count_words(final_summary)

        # Calculate ROUGE
        scores = rouge_metric.compute(predictions=[final_summary], references=[text])
        rouge1 = scores['rouge1']
        rouge2 = scores['rouge2']
        rougeL = scores['rougeL']
        rouge_info = f"""
**ROUGE Scores:**  
ROUGE-1: {rouge1:.3f}  
ROUGE-2: {rouge2:.3f}  
ROUGE-L: {rougeL:.3f}  

**Note:**  
- ROUGE-1 → Word-level overlap  
- ROUGE-2 → Bigram-level overlap  
- ROUGE-L → Longest common subsequence overlap
"""

        # Save to history
        history = load_history()
        history_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input": text[:200] + "..." if len(text) > 200 else text,
            "input_words": word_count,
            "summary": final_summary,
            "summary_words": summary_word_count
        }
        history.append(history_entry)
        save_history(history)

        return {'error': None, 'summary': final_summary, 'word_count': summary_word_count, 'input_words': word_count, 'rouge_info': rouge_info}
    
    except Exception as e:
        return {'error': f'Error generating summary: {str(e)}', 'summary': '', 'word_count': 0, 'rouge_info': ''}

# ----------------------
# GRADIO INTERFACE FUNCTIONS
# ----------------------
def summarize_interface(text):
    result = summarize_text(text)
    if result['error']:
        return result['error'], "", "", get_history()
    word_info = f"Summary: {result['word_count']} words"
    return result['summary'], word_info, result.get('rouge_info', ""), get_history()

def update_word_count(text):
    count = count_words(text)
    min_words = 50
    if count < min_words:
        return f"Words: {count}/{min_words} (need {min_words - count} more)"
    return f"Words: {count}/{min_words} ✓"

def clear_history_interface():
    clear_history()
    return "History cleared!", get_history()

# ----------------------
# SAMPLE TEXTS
# ----------------------
SAMPLE_TEXTS = [
    {
        "title": "Climate Change",
        "content": """Climate change continues to pose significant challenges worldwide as global temperatures rise and weather patterns become increasingly unpredictable. Scientists have observed unprecedented changes in Arctic ice coverage, with some regions experiencing complete ice loss during summer months. The melting of polar ice caps contributes to rising sea levels, threatening coastal communities and low-lying island nations. Extreme weather events, including more frequent and intense hurricanes, droughts, and flooding, have become commonplace across different continents. Governments and international organizations are implementing various strategies to combat these effects, including renewable energy initiatives, carbon taxation, and reforestation programs. However, many experts argue that current efforts are insufficient to meet the targets set by the Paris Climate Agreement. The transition to sustainable energy sources requires substantial investment in infrastructure and technology development. Meanwhile, developing nations face particular challenges in balancing economic growth with environmental protection, often lacking the resources necessary for large-scale green initiatives."""
    },
    {
        "title": "Artificial Intelligence", 
        "content": """Artificial intelligence technology has revolutionized numerous industries and continues to evolve at an unprecedented pace. Machine learning algorithms now power everything from recommendation systems on streaming platforms to autonomous vehicles navigating complex urban environments. Natural language processing has enabled chatbots and virtual assistants to understand and respond to human communication with remarkable accuracy. In healthcare, AI systems assist doctors in diagnosing diseases, analyzing medical imagery, and developing personalized treatment plans for patients. The financial sector utilizes AI for fraud detection, algorithmic trading, and risk assessment, processing vast amounts of data in real-time. Educational institutions are incorporating AI-powered platforms to create personalized learning experiences and provide instant feedback to students. However, the rapid advancement of AI also raises important ethical questions about job displacement, privacy concerns, and the need for regulatory frameworks. Industry leaders emphasize the importance of responsible AI development, ensuring that these powerful technologies benefit society while minimizing potential risks and unintended consequences."""
    },
    {
        "title": "Space Exploration",
        "content": """Space exploration has entered a new era with private companies joining government agencies in ambitious missions beyond Earth's atmosphere. Recent successful launches by commercial space companies have demonstrated the viability of reusable rocket technology, significantly reducing the cost of space travel. The International Space Station continues to serve as a crucial platform for scientific research, hosting experiments in microgravity that advance our understanding of physics, biology, and materials science. Mars exploration missions have provided valuable data about the Red Planet's geology, climate history, and potential for past or present microbial life. Future missions plan to establish permanent human settlements on Mars, requiring innovative solutions for life support systems, radiation protection, and sustainable resource utilization. The development of space tourism opens new possibilities for civilian space travel, though safety regulations and cost considerations remain significant barriers. Satellite technology improvements enable better Earth observation, GPS navigation, and global communications networks. Scientists are also exploring the potential for asteroid mining to access rare minerals and resources that could support both Earth-based industries and future space colonies."""
    }
]

# ----------------------
# GRADIO APP
# ----------------------
with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate", neutral_hue="slate"),
    title="Article Summarizer",
    css="""
    .word-counter { font-size:12px; color:#666; text-align:right; margin-top:5px; }
    .summary-info { font-size:12px; color:#4CAF50; font-weight:bold; margin-bottom:10px; }
    .example-section { border:1px solid #e0e0e0; border-radius:8px; padding:15px; margin:10px 0; background-color:#f9f9f9; }
    """
) as demo:
    
    gr.Markdown("# 📄 Article Summarizer\nGenerate concise summaries of articles using a fine-tuned BART model.")
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 📝 Input Article")
            input_text = gr.Textbox(lines=12, placeholder="Paste your article here (minimum 50 words required)...", show_label=False)
            word_counter = gr.Markdown("Words: 0/50", elem_classes=["word-counter"])
            with gr.Row():
                summarize_btn = gr.Button("🔍 Generate Summary", variant="primary")
                clear_btn = gr.Button("🗑️ Clear")
            
            # Examples
            with gr.Group(elem_classes=["example-section"]):
                gr.Markdown("### 💡 Example Articles")
                example_buttons = []
                for sample in SAMPLE_TEXTS:
                    btn = gr.Button(f"📰 {sample['title']} Article")
                    example_buttons.append((btn, sample['content']))
        
        with gr.Column(scale=2):
            gr.Markdown("### 📋 Generated Summary")
            summary_info = gr.Markdown("", elem_classes=["summary-info"])
            summary_output = gr.Textbox(lines=10, show_label=False, show_copy_button=True, interactive=False)
            
            gr.Markdown("### 📊 ROUGE Scores")
            rouge_display = gr.Markdown("", elem_classes=["summary-info"])
    
    # History
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📚 Recent Summaries")
            with gr.Row():
                refresh_history_btn = gr.Button("🔄 Refresh")
                clear_history_btn = gr.Button("🗑️ Clear History")
            history_display = gr.Markdown(get_history(), max_height=300)
    
    # EVENTS
    input_text.change(fn=update_word_count, inputs=[input_text], outputs=[word_counter])
    summarize_btn.click(
        fn=summarize_interface,
        inputs=[input_text],
        outputs=[summary_output, summary_info, rouge_display, history_display]
    )
    clear_btn.click(fn=lambda: ("", "Words: 0/50", "", ""), outputs=[input_text, word_counter, summary_info, rouge_display])
    
    for btn, example_content in example_buttons:
        btn.click(fn=lambda x=example_content: (x, update_word_count(x)), outputs=[input_text, word_counter])
    
    refresh_history_btn.click(fn=get_history, outputs=[history_display])
    clear_history_btn.click(fn=clear_history_interface, outputs=[summary_info, history_display])
    
    gr.Markdown("---\n💡 **Tips:**\n• Enter at least 50 words\n• Best with news articles\n• History is automatically saved")

# ----------------------
# LAUNCH
# ----------------------
if __name__ == "__main__":
    demo.launch()
