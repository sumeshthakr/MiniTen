"""
Text/NLP Processing Utilities

Text processing for natural language understanding tasks.
Optimized for edge devices with minimal dependencies.

Features:
- Tokenization (word, character, BPE-like)
- Vocabulary management
- Word embeddings
- Text preprocessing
- Sequence padding
- Optimized for on-device NLP
- Minimal external dependencies
"""

import re
import string
from typing import List, Dict, Optional, Tuple, Union
import math


# ============================================================================
# Tokenizers
# ============================================================================

class Tokenizer:
    """
    Base tokenizer class.
    
    Converts text to token IDs and vice versa.
    """
    
    # Special tokens
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    
    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self._init_special_tokens()
    
    def _init_special_tokens(self):
        """Initialize special tokens."""
        special = [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]
        for i, token in enumerate(special):
            self.vocab[token] = i
            self.inverse_vocab[i] = token
    
    @property
    def pad_token_id(self) -> int:
        return self.vocab[self.PAD_TOKEN]
    
    @property
    def unk_token_id(self) -> int:
        return self.vocab[self.UNK_TOKEN]
    
    @property
    def bos_token_id(self) -> int:
        return self.vocab[self.BOS_TOKEN]
    
    @property
    def eos_token_id(self) -> int:
        return self.vocab[self.EOS_TOKEN]
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into tokens.
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens
        """
        raise NotImplementedError("Subclass must implement tokenize()")
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Convert text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add BOS/EOS tokens
            
        Returns:
            List of token IDs
        """
        tokens = self.tokenize(text)
        
        if add_special_tokens:
            tokens = [self.BOS_TOKEN] + tokens + [self.EOS_TOKEN]
        
        return [self.vocab.get(t, self.unk_token_id) for t in tokens]
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Convert token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text string
        """
        special_ids = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        
        tokens = []
        for tid in token_ids:
            if skip_special_tokens and tid in special_ids:
                continue
            token = self.inverse_vocab.get(tid, self.UNK_TOKEN)
            tokens.append(token)
        
        return self._detokenize(tokens)
    
    def _detokenize(self, tokens: List[str]) -> str:
        """Convert tokens back to text."""
        return " ".join(tokens)
    
    def fit(self, texts: List[str]):
        """Build vocabulary from texts."""
        for text in texts:
            for token in self.tokenize(text):
                if token not in self.vocab:
                    idx = len(self.vocab)
                    self.vocab[token] = idx
                    self.inverse_vocab[idx] = token
    
    def __len__(self) -> int:
        return len(self.vocab)


class WordTokenizer(Tokenizer):
    """
    Word-level tokenizer.
    Splits text on whitespace and punctuation.
    """
    
    def __init__(self, lowercase: bool = True, 
                 remove_punctuation: bool = False):
        super().__init__()
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        if self.lowercase:
            text = text.lower()
        
        # Split on whitespace and keep punctuation as separate tokens
        tokens = re.findall(r"\w+|[^\w\s]", text)
        
        if self.remove_punctuation:
            tokens = [t for t in tokens if t not in string.punctuation]
        
        return tokens
    
    def _detokenize(self, tokens: List[str]) -> str:
        """Join tokens back to text, handling punctuation."""
        if not tokens:
            return ""
        
        result = [tokens[0]]
        for token in tokens[1:]:
            if token in string.punctuation:
                result.append(token)
            else:
                result.append(" " + token)
        
        return "".join(result)


class CharTokenizer(Tokenizer):
    """
    Character-level tokenizer.
    """
    
    def __init__(self, lowercase: bool = False):
        super().__init__()
        self.lowercase = lowercase
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into characters."""
        if self.lowercase:
            text = text.lower()
        return list(text)
    
    def _detokenize(self, tokens: List[str]) -> str:
        """Join characters back to text."""
        return "".join(tokens)


class SubwordTokenizer(Tokenizer):
    """
    Simple BPE-like subword tokenizer.
    Learns common character pairs and merges them.
    """
    
    def __init__(self, vocab_size: int = 1000, lowercase: bool = True):
        super().__init__()
        self.target_vocab_size = vocab_size
        self.lowercase = lowercase
        self.merges: List[Tuple[str, str]] = []
    
    def fit(self, texts: List[str]):
        """Learn BPE merges from texts."""
        # Initialize with characters
        word_freqs: Dict[str, int] = {}
        
        for text in texts:
            if self.lowercase:
                text = text.lower()
            words = text.split()
            for word in words:
                # Add end-of-word marker
                word = " ".join(list(word)) + " </w>"
                word_freqs[word] = word_freqs.get(word, 0) + 1
        
        # Add initial characters to vocab
        for word in word_freqs:
            for char in word.split():
                if char not in self.vocab:
                    idx = len(self.vocab)
                    self.vocab[char] = idx
                    self.inverse_vocab[idx] = char
        
        # Learn merges
        while len(self.vocab) < self.target_vocab_size:
            # Count pairs
            pair_freqs: Dict[Tuple[str, str], int] = {}
            for word, freq in word_freqs.items():
                symbols = word.split()
                for i in range(len(symbols) - 1):
                    pair = (symbols[i], symbols[i + 1])
                    pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
            
            if not pair_freqs:
                break
            
            # Find most frequent pair
            best_pair = max(pair_freqs, key=pair_freqs.get)
            
            # Merge pair
            self.merges.append(best_pair)
            merged = best_pair[0] + best_pair[1]
            
            if merged not in self.vocab:
                idx = len(self.vocab)
                self.vocab[merged] = idx
                self.inverse_vocab[idx] = merged
            
            # Update words
            new_word_freqs = {}
            for word, freq in word_freqs.items():
                new_word = word.replace(
                    f"{best_pair[0]} {best_pair[1]}", 
                    merged
                )
                new_word_freqs[new_word] = freq
            word_freqs = new_word_freqs
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using learned merges."""
        if self.lowercase:
            text = text.lower()
        
        tokens = []
        for word in text.split():
            word = " ".join(list(word)) + " </w>"
            
            # Apply merges
            for merge in self.merges:
                word = word.replace(f"{merge[0]} {merge[1]}", merge[0] + merge[1])
            
            tokens.extend(word.split())
        
        return tokens
    
    def _detokenize(self, tokens: List[str]) -> str:
        """Convert subword tokens back to text."""
        text = "".join(tokens)
        text = text.replace("</w>", " ")
        return text.strip()


# ============================================================================
# Vocabulary
# ============================================================================

class Vocabulary:
    """
    Vocabulary for mapping tokens to indices.
    
    Args:
        max_size: Maximum vocabulary size
        min_freq: Minimum frequency for token inclusion
    """
    
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    
    def __init__(self, max_size: Optional[int] = None, min_freq: int = 1):
        self.max_size = max_size
        self.min_freq = min_freq
        self.word2idx: Dict[str, int] = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1
        }
        self.idx2word: Dict[int, str] = {
            0: self.PAD_TOKEN,
            1: self.UNK_TOKEN
        }
        self.word_counts: Dict[str, int] = {}
    
    def build(self, texts: Union[List[str], List[List[str]]]):
        """
        Build vocabulary from texts.
        
        Args:
            texts: List of strings or list of token lists
        """
        # Count words
        for text in texts:
            if isinstance(text, str):
                words = text.split()
            else:
                words = text
            
            for word in words:
                self.word_counts[word] = self.word_counts.get(word, 0) + 1
        
        # Filter by frequency and sort by count
        filtered = [(w, c) for w, c in self.word_counts.items() 
                    if c >= self.min_freq]
        filtered.sort(key=lambda x: -x[1])
        
        # Apply max size
        if self.max_size:
            filtered = filtered[:self.max_size - 2]  # Reserve for PAD, UNK
        
        # Build mappings
        for word, _ in filtered:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
    
    def add_word(self, word: str) -> int:
        """
        Add word to vocabulary.
        
        Returns:
            Index of the word
        """
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            return idx
        return self.word2idx[word]
    
    def get_index(self, word: str) -> int:
        """Get index for word."""
        return self.word2idx.get(word, self.word2idx[self.UNK_TOKEN])
    
    def get_word(self, idx: int) -> str:
        """Get word for index."""
        return self.idx2word.get(idx, self.UNK_TOKEN)
    
    def __len__(self) -> int:
        return len(self.word2idx)
    
    def __contains__(self, word: str) -> bool:
        return word in self.word2idx


# ============================================================================
# Sequence Utilities
# ============================================================================

def pad_sequence(sequences: List[List[int]], 
                 max_length: Optional[int] = None,
                 padding_value: int = 0,
                 padding_side: str = 'right') -> List[List[int]]:
    """
    Pad sequences to same length.
    
    Args:
        sequences: List of sequences (list of ints)
        max_length: Maximum length (default: longest sequence)
        padding_value: Value to use for padding
        padding_side: 'right' or 'left'
        
    Returns:
        Padded sequences
    """
    if max_length is None:
        max_length = max(len(s) for s in sequences) if sequences else 0
    
    padded = []
    for seq in sequences:
        if len(seq) >= max_length:
            padded.append(seq[:max_length])
        else:
            pad_len = max_length - len(seq)
            if padding_side == 'right':
                padded.append(seq + [padding_value] * pad_len)
            else:
                padded.append([padding_value] * pad_len + seq)
    
    return padded


def truncate_sequence(sequence: List[int], 
                      max_length: int,
                      truncation_side: str = 'right') -> List[int]:
    """Truncate sequence to max length."""
    if len(sequence) <= max_length:
        return sequence
    
    if truncation_side == 'right':
        return sequence[:max_length]
    else:
        return sequence[-max_length:]


def create_attention_mask(sequences: List[List[int]], 
                          pad_token_id: int = 0) -> List[List[int]]:
    """
    Create attention mask for padded sequences.
    
    Args:
        sequences: Padded sequences
        pad_token_id: ID of padding token
        
    Returns:
        Attention masks (1 for real tokens, 0 for padding)
    """
    return [[0 if t == pad_token_id else 1 for t in seq] for seq in sequences]


# ============================================================================
# Text Preprocessing
# ============================================================================

def lowercase(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()


def uppercase(text: str) -> str:
    """Convert text to uppercase."""
    return text.upper()


def remove_punctuation(text: str) -> str:
    """Remove punctuation from text."""
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)


def remove_digits(text: str) -> str:
    """Remove digits from text."""
    return re.sub(r'\d+', '', text)


def remove_extra_whitespace(text: str) -> str:
    """Remove extra whitespace."""
    return ' '.join(text.split())


def remove_urls(text: str) -> str:
    """Remove URLs from text."""
    return re.sub(r'https?://\S+|www\.\S+', '', text)


def remove_html_tags(text: str) -> str:
    """Remove HTML tags from text."""
    return re.sub(r'<[^>]+>', '', text)


def remove_emails(text: str) -> str:
    """Remove email addresses from text."""
    return re.sub(r'\S+@\S+', '', text)


# Common English stopwords
ENGLISH_STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
    'to', 'was', 'were', 'will', 'with', 'this', 'but', 'they',
    'have', 'had', 'what', 'when', 'where', 'who', 'which', 'why', 'how',
    'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some',
    'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
    'very', 'can', 'just', 'should', 'now', 'i', 'me', 'my', 'myself', 'we',
    'our', 'you', 'your', 'him', 'his', 'she', 'her', 
    'them', 'their', 'am', 'been', 'being', 'do', 'does', 'did',
    'doing', 'would', 'could', 'might', 'must', 'shall'
}


def remove_stopwords(text: str, 
                     stopwords: Optional[set] = None) -> str:
    """
    Remove common stopwords from text.
    
    Args:
        text: Input text
        stopwords: Set of stopwords (default: English stopwords)
        
    Returns:
        Text with stopwords removed
    """
    if stopwords is None:
        stopwords = ENGLISH_STOPWORDS
    
    words = text.lower().split()
    filtered = [w for w in words if w not in stopwords]
    return ' '.join(filtered)


def normalize_text(text: str,
                   lower: bool = True,
                   remove_punct: bool = True,
                   remove_stops: bool = False,
                   remove_extra_ws: bool = True) -> str:
    """
    Apply common text normalization steps.
    
    Args:
        text: Input text
        lower: Convert to lowercase
        remove_punct: Remove punctuation
        remove_stops: Remove stopwords
        remove_extra_ws: Remove extra whitespace
        
    Returns:
        Normalized text
    """
    if lower:
        text = lowercase(text)
    if remove_punct:
        text = remove_punctuation(text)
    if remove_stops:
        text = remove_stopwords(text)
    if remove_extra_ws:
        text = remove_extra_whitespace(text)
    
    return text


# ============================================================================
# N-grams
# ============================================================================

def get_ngrams(text: str, n: int = 2, 
               word_level: bool = True) -> List[Tuple[str, ...]]:
    """
    Extract n-grams from text.
    
    Args:
        text: Input text
        n: Size of n-grams
        word_level: If True, word n-grams. If False, character n-grams.
        
    Returns:
        List of n-gram tuples
    """
    if word_level:
        tokens = text.split()
    else:
        tokens = list(text)
    
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngrams.append(tuple(tokens[i:i + n]))
    
    return ngrams


def get_skipgrams(text: str, n: int = 2, k: int = 1,
                  word_level: bool = True) -> List[Tuple[str, ...]]:
    """
    Extract skip-grams from text.
    
    Args:
        text: Input text
        n: Size of n-grams
        k: Number of words that can be skipped
        word_level: If True, word-level. If False, character-level.
        
    Returns:
        List of skip-gram tuples
    """
    if word_level:
        tokens = text.split()
    else:
        tokens = list(text)
    
    skipgrams = []
    for i in range(len(tokens)):
        for j in range(i + 1, min(i + k + 2, len(tokens))):
            if n == 2:
                skipgrams.append((tokens[i], tokens[j]))
    
    return skipgrams


# ============================================================================
# Word Embeddings
# ============================================================================

class SimpleEmbedding:
    """
    Simple word embedding lookup table.
    For edge devices - uses lists instead of numpy.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Initialize with random values (Xavier-like)
        import random
        scale = math.sqrt(6.0 / (vocab_size + embedding_dim))
        self.weights = [
            [random.uniform(-scale, scale) for _ in range(embedding_dim)]
            for _ in range(vocab_size)
        ]
    
    def __call__(self, token_ids: List[int]) -> List[List[float]]:
        """
        Look up embeddings for token IDs.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            List of embedding vectors
        """
        return [self.weights[tid] if tid < self.vocab_size 
                else [0.0] * self.embedding_dim
                for tid in token_ids]
    
    def from_pretrained(self, embeddings: Dict[str, List[float]], 
                        vocab: Dict[str, int]):
        """Load pretrained embeddings."""
        for word, idx in vocab.items():
            if word in embeddings and idx < self.vocab_size:
                self.weights[idx] = embeddings[word]


# ============================================================================
# Text Similarity
# ============================================================================

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute Levenshtein (edit) distance between two strings.
    
    Args:
        s1, s2: Input strings
        
    Returns:
        Edit distance
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = list(range(len(s2) + 1))
    
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def jaccard_similarity(text1: str, text2: str, 
                       word_level: bool = True) -> float:
    """
    Compute Jaccard similarity between two texts.
    
    Args:
        text1, text2: Input texts
        word_level: If True, word-level. If False, character-level.
        
    Returns:
        Jaccard similarity (0 to 1)
    """
    if word_level:
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
    else:
        set1 = set(text1.lower())
        set2 = set(text2.lower())
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0


def cosine_similarity_vectors(v1: List[float], v2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot / (norm1 * norm2)


def bag_of_words(texts: List[str], 
                 vocab: Optional[Vocabulary] = None) -> Tuple[List[List[int]], Vocabulary]:
    """
    Convert texts to bag-of-words representation.
    
    Args:
        texts: List of texts
        vocab: Optional vocabulary (built if not provided)
        
    Returns:
        Tuple of (bow vectors, vocabulary)
    """
    if vocab is None:
        vocab = Vocabulary()
        vocab.build(texts)
    
    bow_vectors = []
    for text in texts:
        vector = [0] * len(vocab)
        for word in text.lower().split():
            idx = vocab.get_index(word)
            if idx < len(vector):
                vector[idx] += 1
        bow_vectors.append(vector)
    
    return bow_vectors, vocab


def tfidf(texts: List[str], 
          vocab: Optional[Vocabulary] = None) -> Tuple[List[List[float]], Vocabulary]:
    """
    Compute TF-IDF representation for texts.
    
    Args:
        texts: List of texts
        vocab: Optional vocabulary
        
    Returns:
        Tuple of (TF-IDF vectors, vocabulary)
    """
    if vocab is None:
        vocab = Vocabulary()
        vocab.build(texts)
    
    n_docs = len(texts)
    
    # Compute document frequencies
    doc_freq = [0] * len(vocab)
    for text in texts:
        seen = set()
        for word in text.lower().split():
            idx = vocab.get_index(word)
            if idx not in seen and idx < len(doc_freq):
                doc_freq[idx] += 1
                seen.add(idx)
    
    # Compute IDF
    idf = [math.log(n_docs / (df + 1)) + 1 if df > 0 else 0 
           for df in doc_freq]
    
    # Compute TF-IDF vectors
    tfidf_vectors = []
    for text in texts:
        # Term frequency
        tf = [0] * len(vocab)
        words = text.lower().split()
        for word in words:
            idx = vocab.get_index(word)
            if idx < len(tf):
                tf[idx] += 1
        
        # Normalize TF
        max_tf = max(tf) if tf else 1
        tf = [t / max_tf if max_tf > 0 else 0 for t in tf]
        
        # TF-IDF
        vector = [t * i for t, i in zip(tf, idf)]
        tfidf_vectors.append(vector)
    
    return tfidf_vectors, vocab
