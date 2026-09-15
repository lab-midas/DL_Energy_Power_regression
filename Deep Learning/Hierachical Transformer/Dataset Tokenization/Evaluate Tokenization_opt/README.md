# 🧩 Tokenizer Exploration

Comprehensive evaluation and optimization of tokenization strategies for MRI text data processing.

---

## 🎯 Objectives

- 🚀 **Strategy Comparison:** Systematic evaluation of popular pretrained tokenizers and preprocessing approaches
- 📝 **Domain Adaptation:** Custom tokenizer training on the complete MRI text dataset for optimal performance
- 🏆 **Performance Optimization:** Comparative analysis to identify the most effective tokenization method for this specific domain

## 📋 Implementation

- **🔬 Tokenizer Evaluation:** Comprehensive testing of three major tokenization approaches:
  - **BPE (Byte-Pair Encoding):** Standard subword tokenization method
  - **WordPiece:** Google's tokenization algorithm used in BERT
  - **Unigram:** Probabilistic subword tokenization approach
- **📊 Training Data:** `train_text.txt` contains the complete MRI text corpus including:
  - BodyRegion_exam, Program_exam, LeanExamination_exam
  - BodyRegion_meas, Protocol_meas, LeanProtocol_meas, Coils_meas
- **🛠️ Preprocessing Pipeline:** Advanced text normalization and special token handling:
  - NaN values were replaced with a special `[NAN]` token to enable proper <br>
    tokenization and consistent sequence lengths.
  - Special characters were normalized before tokenization: <br>
    `,` or `;` with [SEP], `_`, `%\`, `%/`, `%` with [SUBSEQ], `/` with [SUB], and <br>
    `+` with [AND]
  - Numbered bracketed patterns like `[/12]` or `[3]` were converted to <br>
    `[REPEAT]12` or `[REPEAT]3` so repeated protocol fragments can be <br>
    recognized consistently.
  - Text normalization with lowercase conversion and accent removal
  - Strategic pattern splitting for MRI-specific terminology


Replace the  in the text columns to create special tokens that can be used to

## 🏆 Results

**WordPiece Tokenizer** emerged as the optimal solution for this MRI domain:
- Superior handling of medical terminology and technical sequences
- Effective subword segmentation for MRI-specific compound terms
- Best balance between vocabulary size and semantic preservation
- Robust performance on domain-specific special tokens and patterns

The trained WordPiece tokenizer is saved as `tokenizer_wordpiece.json` and <br>
integrated into the multimodal dataset pipeline.