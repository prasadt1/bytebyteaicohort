# LLM Playground Notebook - Complete Beginner's Guide

## What Is This Notebook About?

This notebook is a hands-on introduction to **Large Language Models (LLMs)** - the AI systems that power tools like ChatGPT. It teaches you the fundamental concepts behind how these models work by walking you through practical code examples. Think of it as opening up the hood of a car to understand how the engine works!

The notebook is designed to run on **Google Colab**, which is a free online platform where you can run Python code without installing anything on your computer.

---

## Overall Learning Goals

By the end of this notebook, you'll understand:

1. **Tokenization** - How text is converted into numbers that AI can process
2. **GPT-2 & Transformer Architecture** - The building blocks of modern language models
3. **Loading Pre-trained Models** - How to use existing AI models with Hugging Face
4. **Text Generation Strategies** - Different ways AI can generate text
5. **Model Types** - The difference between completion models and instruction-following models

---

## Section-by-Section Breakdown

### Section 0: Setup and Imports

**What's happening:**
```python
import torch, transformers, tiktoken
```

**Explanation for beginners:**
- This imports the necessary libraries (tools) we need:
  - **torch (PyTorch)**: The framework that runs neural networks
  - **transformers**: Hugging Face's library with pre-built AI models
  - **tiktoken**: OpenAI's tokenization library

Think of these like importing different toolboxes before starting a project.

---

## Section 1: Tokenization

### Why Tokenization Matters

**The Problem:** Computers can't understand text directly. They only understand numbers.

**The Solution:** Tokenization converts text into numerical IDs that neural networks can process.

**Real-world analogy:** It's like translating a book from English to a secret numeric code where each word or part of a word gets a unique number.

---

### 1.1 Word-Level Tokenization

**What's happening:**
The notebook creates a simple word-level tokenizer from scratch.

**Step-by-step breakdown:**

```python
# 1. Create a mini corpus (collection of text)
corpus = [
    "The quick brown fox jumps over the lazy dog",
    "Tokenization converts text to numbers",
    "Large language models predict the next token"
]
```
- This creates three example sentences to work with

```python
# 2. Build the vocabulary
vocab = []
PAD, UNK = "[PAD]", "[UNK]"
```
- Creates an empty vocabulary list
- Defines two special tokens:
  - **[PAD]**: Used to make sentences the same length
  - **[UNK]**: Used for unknown words not in the vocabulary

```python
# Extract unique words from all sentences
for sentence in corpus:
    set_from_corpus.update(sentence.lower().split(" "))

vocab = list(set_from_corpus)
vocab.insert(0, PAD)
vocab.insert(1, UNK)
```
- Splits each sentence into individual words
- Converts to lowercase (so "The" and "the" are treated the same)
- Creates a unique list of all words
- Adds special tokens at the beginning

```python
# Create dictionaries for converting between words and IDs
for index, word in enumerate(vocab):
    word2id[word] = index
    id2word[index] = word
```
- Creates two lookup tables:
  - **word2id**: Converts words to numbers (e.g., "the" → 11)
  - **id2word**: Converts numbers back to words (e.g., 11 → "the")

```python
def encode(text):
    words = text.split(" ")
    ids = [word2id.get(word, 1) for word in words]
    return ids
```
- **encode()**: Takes text and converts it to a list of numbers
- If a word isn't found, it uses 1 (which is [UNK])

```python
def decode(ids):
    words = [id2word.get(id, "[UNK]") for id in ids]
    decoded_text = " ".join(words)
    return decoded_text
```
- **decode()**: Takes a list of numbers and converts back to text

**Example output:**
```
Input text : The brown horse jumps
Token IDs  : [1, 18, 1, 20]
Decoded    : [UNK] brown [UNK] jumps
```

**Why [UNK] appears:** "The" and "horse" weren't in the original three-sentence corpus, so they're marked as unknown.

**Limitations of word-level tokenization:**
1. **Huge vocabulary**: You'd need millions of entries for all English words
2. **Unknown words**: Any new or misspelled word becomes [UNK]

---

### 1.2 Character-Level Tokenization

**What's happening:**
Instead of treating whole words as tokens, every single character (letter, space, punctuation) gets its own ID.

**How it works:**
```python
import string
charset = sorted(list(set(string.ascii_letters)))
vocab = ["[PAD]", "[UNK]"] + charset
```
- Creates a vocabulary of all letters (a-z, A-Z) plus special tokens

```python
def encode(text):
    return [char2id.get(c, 1) for c in text]
```
- Converts each character to its ID

**Example:**
```
Input text : Hello prasad how are you doing?
Token IDs  : [18, 15, 22, 22, 25, 1, 26, 28, ...]
Decoded    : Hello prasad how are you doing?
```

**Pros:**
- No unknown characters (unless you use emojis or special symbols)
- Very small vocabulary (only ~50-100 tokens)

**Cons:**
- Creates VERY long sequences (every character is separate)
- Model has to learn how to combine characters into meaningful words

---

### 1.3 Subword-Level Tokenization (BPE)

**What's happening:**
This is the "Goldilocks" solution used by modern LLMs like GPT. It splits text into subword pieces that are smaller than words but larger than characters.

**Key concept: Byte Pair Encoding (BPE)**
- Frequently occurring character combinations become single tokens
- Less common words get split into smaller pieces

**Example with tiktoken (OpenAI's tokenizer):**
```python
enc = tiktoken.get_encoding("gpt2")
text = "Hello prasad how are you doing?"
ids = enc.encode(text)
```

**Output:**
```
Token IDs  : [15496, 778, 292, 324, 703, 389, 345, 1804, 30]
Tokens     : ['Hello', ' pras', 'ad', ' how', ' are', ' you', ' doing', '?']
```

**Notice:**
- "Hello" is one token (common word)
- "prasad" is split into " pras" + "ad" (less common name)
- Spaces are included with words (indicated by '·')

**Why this is best:**
- Moderate vocabulary size (~50,000 tokens for GPT-2)
- No unknown words (can represent any text)
- Efficient sequence lengths

---

## Section 2: Understanding GPT-2

### 2.1 Loading GPT-2

**What's happening:**
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
```

**Explanation:**
- Downloads a pre-trained GPT-2 model from Hugging Face
- GPT-2 is a smaller version of the technology behind ChatGPT
- It has 124 million parameters (think of these as the model's "knowledge")

**What "pre-trained" means:**
The model has already learned patterns from billions of words of text from the internet. You don't have to train it yourself!

---

### 2.2 Model Architecture

**What's happening:**
```python
print(model)
```

**Output breakdown:**

```
GPT2LMHeadModel(
  (transformer): GPT2Model(
    (wte): Embedding(50257, 768)
```
- **Embedding layer**: Converts token IDs into 768-dimensional vectors (mathematical representations)
- Vocabulary size: 50,257 tokens

```
    (wpe): Embedding(1024, 768)
```
- **Position embeddings**: Helps the model understand the order of words
- Can handle up to 1024 tokens at once

```
    (h): ModuleList(
      (0-11): 12 x GPT2Block(
```
- **12 transformer blocks**: The core processing units
- Each block has:
  - **Self-attention**: Helps the model understand relationships between words
  - **Feed-forward network**: Processes the information
  - **Layer normalization**: Keeps values stable during processing

```
    (ln_f): LayerNorm()
  )
  (lm_head): Linear(in_features=768, out_features=50257)
)
```
- **Final layer**: Converts the processed information back into probabilities for each possible next token

**Analogy:** Think of the transformer blocks like layers of filters in photo editing - each layer refines and processes the information differently.

---

### 2.3 Model Statistics

**What's happening:**
```python
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
```

**Output:**
```
Total parameters: 124,439,808
```

**What this means:**
- The model has 124 million adjustable numbers (parameters)
- These were learned during training on massive amounts of text
- More parameters generally means more capable, but also slower and more memory-intensive

**Memory calculation:**
```
Model size: ~498 MB
```
- Each parameter stored as 32-bit float takes 4 bytes
- 124M parameters × 4 bytes ≈ 498 MB

---

## Section 3: Text Generation

### 3.1 Simple Generation

**What's happening:**
```python
prompt = "The future of artificial intelligence is"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
output = model.generate(input_ids, max_length=50)
generated_text = tokenizer.decode(output[0])
```

**Step-by-step:**
1. **Tokenize the prompt**: Convert text to token IDs
2. **Generate**: Model predicts next tokens one by one
3. **Decode**: Convert token IDs back to readable text

**Example output:**
```
The future of artificial intelligence is not just about the technology itself, but about how we use it to improve our lives.
```

---

### 3.2 Decoding Strategies

The notebook explores different ways to choose the next token. This is crucial because it dramatically affects the quality and creativity of generated text.

#### Strategy 1: Greedy Decoding

**What it does:**
Always picks the single most likely next token.

**Code:**
```python
output = model.generate(
    input_ids,
    max_length=50,
    do_sample=False  # No randomness
)
```

**Pros:**
- Fast and deterministic (same input = same output)
- Coherent for simple completions

**Cons:**
- Repetitive and boring
- Can get stuck in loops
- No creativity

**Example:**
```
"I really enjoyed the movie because it was very good and I really enjoyed it because it was very good..."
```

---

#### Strategy 2: Sampling with Temperature

**What it does:**
Introduces controlled randomness by sampling from probability distribution.

**Code:**
```python
output = model.generate(
    input_ids,
    max_length=50,
    do_sample=True,
    temperature=0.7  # Controls randomness
)
```

**Temperature explained:**
- **Temperature = 0.0**: Acts like greedy (picks most likely)
- **Temperature = 0.7**: Balanced randomness (recommended)
- **Temperature = 1.0**: Full probability distribution
- **Temperature = 2.0**: Very random, often nonsensical

**Analogy:** Think of temperature like a creativity dial:
- Low temperature = conservative writer
- High temperature = wild, experimental writer

---

#### Strategy 3: Top-K Sampling

**What it does:**
Only considers the K most likely next tokens.

**Code:**
```python
output = model.generate(
    input_ids,
    max_length=50,
    do_sample=True,
    top_k=50  # Only consider top 50 tokens
)
```

**Why it's useful:**
- Filters out very unlikely tokens
- Maintains quality while adding variety
- Prevents random nonsense words

**Example:** If the next word is likely "cat" or "dog", it won't randomly pick "xylophone"

---

#### Strategy 4: Top-P (Nucleus) Sampling

**What it does:**
Considers the smallest set of tokens whose cumulative probability exceeds P.

**Code:**
```python
output = model.generate(
    input_ids,
    max_length=50,
    do_sample=True,
    top_p=0.9  # Consider tokens making up 90% probability
)
```

**How it works:**
1. Sort all tokens by probability
2. Add probabilities from highest to lowest
3. Stop when you reach 90% (or whatever top_p value)
4. Only sample from those tokens

**Why it's better than top-k:**
- Adapts to context dynamically
- When model is confident: considers few tokens
- When model is uncertain: considers many tokens

**This is the most commonly used strategy in modern LLMs!**

---

#### Strategy 5: Beam Search

**What it does:**
Keeps track of multiple possible sequences simultaneously and picks the overall best one.

**Code:**
```python
output = model.generate(
    input_ids,
    max_length=50,
    num_beams=5,  # Keep 5 candidate sequences
    early_stopping=True
)
```

**Analogy:** Like exploring multiple paths in a maze simultaneously, then picking the one that leads to the best destination.

**Pros:**
- Higher quality for tasks like translation
- More coherent overall sequences

**Cons:**
- Slower (computing multiple sequences)
- Can still be repetitive
- Less creative than sampling methods

---

## Section 4: Instruction-Tuned Models

### 4.1 Base vs. Instruction-Tuned Models

**Base models (like GPT-2):**
- Trained to predict the next word
- Good at continuing text
- Don't follow instructions well

**Instruction-tuned models:**
- Further trained on instruction-response pairs
- Designed to follow user commands
- Examples: ChatGPT, Claude, Llama-2-Chat

### 4.2 Using an Instruction-Tuned Model

**What's happening:**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
```

**Special prompt format:**
```python
prompt = """[INST] <<SYS>>
You are a helpful AI assistant.
<</SYS>>

Write a short poem about Python programming. [/INST]"""
```

**Why the special format?**
Instruction-tuned models are trained with specific markers:
- `[INST]`: Start of instruction
- `<<SYS>>`: System message (sets behavior)
- `[/INST]`: End of instruction (model starts response here)

**Example output:**
```
In loops and functions, logic flows,
With Python's syntax, knowledge grows.
From data science to web design,
This language makes solutions shine.
```

---

## Key Takeaways

### 1. Tokenization is Essential
- Converts text to numbers for neural networks
- Subword tokenization (BPE) is the modern standard
- Vocabulary size affects model efficiency

### 2. Transformer Architecture
- Uses self-attention to understand context
- Multiple layers process information
- Parameters store learned knowledge

### 3. Generation Strategies Matter
- Greedy: Fast but repetitive
- Temperature: Controls creativity
- Top-P: Best balance (most commonly used)
- Beam Search: Best for translation/summarization

### 4. Model Types
- **Base models**: Good at text continuation
- **Instruction-tuned**: Good at following commands
- Always use the right model for your task!

---

## Practical Tips for Beginners

1. **Start with smaller models:** GPT-2 is great for learning; save larger models for production

2. **Experiment with parameters:**
   - Try different temperatures (0.5 to 1.0)
   - Adjust max_length based on needs
   - Use top_p=0.9 as a starting point

3. **Watch for repetition:** If output repeats, try:
   - Adding `repetition_penalty=1.2`
   - Using sampling instead of greedy
   - Increasing temperature

4. **Memory management:** Large models need lots of RAM/VRAM:
   - Use Google Colab for free GPU access
   - Consider smaller models for experimentation
   - Use `model.to('cpu')` if GPU runs out of memory

5. **Prompt engineering matters:**
   - Clear, specific prompts get better results
   - Include examples when possible
   - For instruction models, use correct format

---

## Common Issues and Solutions

**Problem:** Model output is nonsensical
- **Solution:** Lower temperature (try 0.7)
- Check if using correct model format

**Problem:** Output is too repetitive
- **Solution:** Add `repetition_penalty=1.2`
- Use sampling instead of greedy
- Try top-p sampling

**Problem:** Out of memory error
- **Solution:** Reduce max_length
- Use smaller model
- Use Colab with GPU runtime

**Problem:** Generation is too slow
- **Solution:** Reduce max_length
- Use greedy instead of beam search
- Enable GPU acceleration

---

## Next Steps

After completing this notebook, you can:

1. **Experiment with different models** on Hugging Face
2. **Try fine-tuning** a model on your own data
3. **Build applications** using these concepts
4. **Explore prompt engineering** techniques
5. **Learn about model evaluation** metrics

---

## Glossary

- **Token**: A piece of text (word, subword, or character) represented as a number
- **Embedding**: Mathematical representation of a token in high-dimensional space
- **Parameter**: Adjustable number in the model learned during training
- **Inference**: Using a trained model to generate predictions
- **Context window**: Maximum number of tokens the model can process at once
- **Attention**: Mechanism that helps model focus on relevant parts of input
- **Fine-tuning**: Further training a pre-trained model on specific data
- **Temperature**: Parameter controlling randomness in text generation
- **Perplexity**: Metric measuring how well model predicts text (lower is better)

---

## Resources for Further Learning

1. **Hugging Face Documentation**: https://huggingface.co/docs
2. **The Illustrated Transformer**: http://jalammar.github.io/illustrated-transformer/
3. **OpenAI Tokenizer Tool**: https://platform.openai.com/tokenizer
4. **Stanford CS224N**: Natural Language Processing with Deep Learning course

---

This notebook provides a solid foundation for understanding how modern language models work. The concepts you've learned here are the building blocks for more advanced topics in AI and NLP!
