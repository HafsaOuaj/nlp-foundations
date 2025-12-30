# Tokenization: from text → numbers

![tokenization_pipeline-dark.svg](attachment:f2ad2ce0-9261-4314-bb0a-29ac153d9ff1:tokenization_pipeline-dark.svg)

- ***Steps: Text —> split into tokens —> token ↔ ids  (vocabulary)—> ids = inputs of language model***
- ***Requirement: handle No/English emojis unseen vocables (optional) characters***
- ***Design Components: Normalizer (remove ‘ (ù → u) lowercase), Pre-tokenizer (split by spaces (avoid long tokens, handle language specifications)), Model, Post-process (adding strings or templates), Decoder (reverse from ids → tokens)***
- Slow tokenizers are those written in Python inside the→ 🤗 Transformers library, while the fast versions are the ones provided by 🤗 Tokenizers, which are written in Rust.
- output of tokenizer is BatchEncoding subclass of dict that behaves as dict stores ids and extra functionality to track the transformations —> padding/truncation and batch
- Out of Vocabulary words are handled by iterative decomposition into small known tokens until convergence.  ***byte-level BPE***
    
    ![image.png](attachment:c66ccc54-032f-459c-9a97-fa4be255db65:image.png)
    
- The score for the WordPiece tokenzers

![image.png](attachment:a7908e2e-8572-48cf-bca3-bfcb2d3c2e98:image.png)

- at token level classification we can use aggregation strategies like first average max

> The softmax doesn’t change the order of the logits so we can preform the argmax on them directly
when defining model define the configs
> 

| **Model** | **BPE** | **WordPiece** | **Unigram** |
| --- | --- | --- | --- |
| Training | Starts from a small vocabulary and learns rules to merge tokens | Starts from a small vocabulary and learns rules to merge tokens | Starts from a large vocabulary and learns rules to remove tokens |
| Training step | Merges the tokens corresponding to the most common pair | Merges the tokens corresponding to the pair with the best score based on the frequency of the pair, privileging pairs where each individual token is less frequent | Removes all the tokens in the vocabulary that will minimize the loss computed on the whole corpus

***It  made the assumption of P(sentence) = Product(P s|s in senence)*** |
| Learns | Merge rules and a vocabulary | Just a vocabulary | A vocabulary with a score for each token |
| Encoding | Splits a word into characters and applies the merges learned during training | Finds the longest subword starting from the beginning that is in the vocabulary, then does the same for the rest of the word | Finds the most likely split into tokens, using the scores learned during training |
|  |  |  |  |

# Masking

- When:
    - Causal Masking → prevent information leakage to the model during the training (prevent the model to have access to future tokens)
    - Padding Masking —> dealing with different lengths sequences, we add tokens (padding) so we don’t want the model to access this values
    - Customer Masking → prevent attention to specific token positions based on the domain application
- To make masked elements contribute 0 to the softmax, you add −∞ to those positions.