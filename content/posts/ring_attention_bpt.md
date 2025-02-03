+++
title = "Handling long context: Understanding concept of Blockwise Parallel Transformers and Ring Attention"
date = 2025-02-03T01:09:05+05:30
tags = ["AI", "Machine Learning", "NLP", "Transformers", "LLM"]
categories = ["Technology", "AI"]
author = "Aadish Agrawal"
description = "Deep dive into concept of multi-head attention, Blockwise Parallel Transformers and Ring Attention."
+++

## Basics of transformer architecture
![Diagram 1: The encoder-decoder structure of the Transformer architecture - Taken from "Attention Is All You Need"](https://raw.githubusercontent.com/aadishagr/hokagehub.github.io/refs/heads/main/content/docs/assets/images/ringatten/transformer.png#center)

The fundamental idea of transformer architecture revolves around positional encoding and the multi-head self-attention mechanism. It includes an encoder, where self-attention is calculated first, followed by a feed-forward layer. The decoder consists of masked multi-head self-attention, then a second layer of multi-head attention, and finally, a feed-forward layer.

The multi-head self-attention mechanism in the encoder determines the relevance of each token within the context. Conversely, the masked multi-head self-attention in the decoder establishes the relevance of each token in a causal manner, meaning the model cannot consider future tokens. The second attention layer in the decoder takes the key and value from the encoder and the query from the previous attention layer of the decoder.

## Positional Encoding
![Diagram 2 :Positional Encoder - Taken from https://github.com/hkproj/transformer-from-scratch-notes](https://raw.githubusercontent.com/aadishagr/hokagehub.github.io/refs/heads/main/content/docs/assets/images/ringatten/positional_encoding.png#center)

Positional encoding provides information about the position of each token within the context. Sinusoidal methods are the preferred algorithms for positional encoding. In Diagram 2, each token is encoded into an embedding vector of size 512. A positional embedding of the same size is then added to each token embedding. These positional encoding vectors are constant and can be reused for different sentences.

## Attention Mechanism
![Diagram 3: Transformer Encoder and Decoder](https://raw.githubusercontent.com/aadishagr/hokagehub.github.io/refs/heads/main/content/docs/assets/images/ringatten/attention.png#center)
![Diagram 4: Mathematical equation for attention calculation](https://raw.githubusercontent.com/aadishagr/hokagehub.github.io/refs/heads/main/content/docs/assets/images/ringatten/qkv.png#center)

The input (X) to the attention layer has a size of (s, d), where s represents the sequence length and d represents the embedding dimension. The Query (Q), Key (K), and Value (V) are projected vectors, with the input of the same dimension (s, d) passing through three linear layers (one each for Q, K, and V). The softmax function is applied to the product of Q and the transpose of K, normalized over the embedding dimension, to provide attention weights, which require memory resources of O(n²). These attention weights are then multiplied by the value matrix to produce the final attention output.

## Multi-Head Attention
![Diagram 5: Multi-Head attention - Taken from https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853](https://raw.githubusercontent.com/aadishagr/hokagehub.github.io/refs/heads/main/content/docs/assets/images/ringatten/multihead.png#center)

The input to the multi-head attention layer has dimensions (s, d), where s is the sequence length and d is the embedding dimension. The first step is to choose the number of attention heads, which determines how the matrix is split. For this example, let's assume there are 2 heads (h), a sequence length of 3 (s), and an embedding dimension of 6 (d). The initial step involves obtaining the linear projection of the input x by passing it through three different linear layers with weight matrices Wq, Wk, and Wv. The data is logically split across the attention heads to enhance computational performance. This means a single data matrix is used for the each matrix of Query, Key, and Value, with logically separate sections of the matrix allocated for each attention head. This approach allows the model to process different parts of the input simultaneously, improving efficiency and capturing more nuanced relationships within the data.

### **Linear Layer logical split**
All the attention heads share the same linear layer but operate on their own logical section of the data matrix. The weights are partitioned based on the number of heads. For example, if the weight matrix is of size d×d, where d is the embedding dimension, the logical partition size is determined by the number of heads.

Query Size(m) = Embedding Size(d) / Number of heads(h)
Then the logical weight for each head of size (d, m).
![Diagram 6: Linear weight logical split - Taken from https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853](https://raw.githubusercontent.com/aadishagr/hokagehub.github.io/refs/heads/main/content/docs/assets/images/ringatten/linear_weight_logical_split.png#center)

### **Q-K-V Matrix logical split**
![Diagram 7: Logical Q matrix split - Taken from https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853](https://raw.githubusercontent.com/aadishagr/hokagehub.github.io/refs/heads/main/content/docs/assets/images/ringatten/logical_q_matrix_split.png#center)

The query matrix, with dimensions (s, d), is logically split to include only the head dimension. Each slice of the original query matrix corresponds to a specific head. By reshaping and splitting, we obtain two slices of the query matrix for each head. Similarly, the key (K) and value (V) matrices are also sliced in this manner. As a result, each head contains the entire sequence, but the embedding dimension is divided based on the number of heads.

### **Multi-Head Attention Computation**
![Diagram 8: Multi-Head Attention Computation - Taken from https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853](https://raw.githubusercontent.com/aadishagr/hokagehub.github.io/refs/heads/main/content/docs/assets/images/ringatten/multi_head_attention_computation.png#center)

For each head, attention is computed using the sliced Q, K, and V matrices to obtain the attention output for each head. This involves a single matrix multiplication, rather than looping through each head. Finally, the outputs from all heads are concatenated to produce the final attention output.

Different parts of the output Embedding can understand various aspects of each word's meaning in relation to other words in the sequence. This enables the Transformer to grasp more nuanced interpretations of the sequence. For instance, one section might capture the 'gender-ness' (male, female, neuter) of a noun while another might capture the 'cardinality' (singular vs plural) of a noun.

## Limitation of Self Attention
Due to the memory requirement for attention computation being O(n²), it increases quadratically with the sequence length. This significant memory demand limits the transformer model's ability to tackle various AI challenges, such as processing videos, high-resolution images, podcasts, code, or books, both during training and inference.
Various techniques have been proposed to reduce the memory requirements of Transformers, including sparse approximation, low-rank approximation, and low-precision approximation. A distinct approach involves computing the softmax matrix in self-attention with linear memory requirements, which can be achieved without materializing the full matrix.

### **Materialising full matrix**
Instead of performing matrix multiplication in a single step, we split the matrix into smaller sections. This allows us to compute the results for each sliced matrix separately and then concatenate them to form the final result. This approach is similar to the techniques used in flash attention, where the full matrix is not materialized all at once, thereby optimizing memory usage.
![Diagram 9: Flash attention](https://raw.githubusercontent.com/aadishagr/hokagehub.github.io/refs/heads/main/content/docs/assets/images/ringatten/flash_attention.png#center)

Flash attention efficiently computes attention in a blockwise manner, but challenges arise with the feed-forward layer. This layer contains a large number of parameters and generates high-dimensional intermediate output vectors, leading to significant memory requirements.
This led to the creation of the Blockwise Parallel Transformer, where both attention computation and the feed-forward network are executed in a blockwise fashion. This approach not only enhances efficiency but also optimizes memory usage, making it a powerful solution for handling complex AI tasks.

## Blockwise Parallel Transformer
The Blockwise Parallel Transformer (BPT) computes both attention and the feed-forward layer in a blockwise manner, significantly reducing memory requirements. This process employs two nested loops to handle input sequence blocks. The outer loop iterates through each block to compute the query, while the inner loop processes each block to generate the key and value. These key-value pairs, along with the query, facilitate the computation of blockwise attention for the corresponding input block. The resulting blockwise attention is then used to generate the output through a feed-forward network, followed by a residual connection. This method enables efficient processing of longer sequences with lower memory usage.
BPT significantly reduces the memory requirements of Transformers, allow to train sequences that are 32 times longer than those handled by vanilla attention and up to 4 times longer than those managed by state-of-the-art Flash Attention models.
![Diagram 10: Blockwise Parallel Transformer Algorithm](https://raw.githubusercontent.com/aadishagr/hokagehub.github.io/refs/heads/main/content/docs/assets/images/ringatten/bpt_algo.png#center)

Self-attention can be computed in a blockwise manner without materializing the softmax attention matrix, softmax(QK^T). The query (Q) is divided into Bq blocks, while the key and value (K, V) are divided into Bkv blocks. For each query block, blockwise attention is computed by iterating over all key-value blocks to obtain local attention. Once the blockwise attention is computed, the global attention matrix is obtained by scaling the blockwise attention.
For a specific query block Qi, 1 ≤ i ≤ Bq, the corresponding attention output can be computed by scaling each blockwise attention as follows:
![Equation - blockwise attention scaling](https://raw.githubusercontent.com/aadishagr/hokagehub.github.io/refs/heads/main/content/docs/assets/images/ringatten/blockwise_attention_scaling.png#center)

This blockwise self-attention computation removes the necessity to materialize the full attention matrix of size O(n²), leading to substantial memory savings. These advancements have reduced the memory overhead of attention to 2bsh bytes per layer, where b is the batch size, s is the sequence length, and h is the hidden size of the model.
Blockwise computation extends beyond self-attention and can be applied to the feed-forward network as well. For each query block, after iterating over the key and value blocks, the feed-forward network is computed along with a residual connection, completing the attention and feed-forward network computation for that query block. This approach means the model processes the feed-forward network on intermediate blocks rather than the entire sequence, resulting in memory savings. The computation for a query block is as follows:
![Equation - FFN](https://raw.githubusercontent.com/aadishagr/hokagehub.github.io/refs/heads/main/content/docs/assets/images/ringatten/ffn.png#center)

### **Limitation of BPT**
While BPT significantly reduces memory demands in Transformers, it still faces a major challenge when scaling up context length due to the need to store the output of each layer. This storage is essential because self-attention inherently involves interactions among all elements (n-to-n interactions). For instance, processing 100 million tokens with a batch size of 1 requires over 1000GB of memory, even for a modest model with a hidden size of 1024. In contrast, modern GPUs and TPUs typically offer less than 100GB of high-bandwidth memory (HBM).

## Ring Attention
![Diagram 11: Ring Attention](https://raw.githubusercontent.com/aadishagr/hokagehub.github.io/refs/heads/main/content/docs/assets/images/ringatten/ring_attention.png#center)

Ring attention enhances the blockwise parallel transformers (BPT) framework by optimizing how input sequences are distributed across different hosts. Each host handles a specific block, running the outer loop of blockwise attention and the corresponding feedforward network. The self-attention mechanism between a query block and a group of key-value blocks is permutation invariant, meaning the attention can be computed in any order. The key is to correctly combine the statistics of each block for accurate rescaling.
![Diagram 12: Ring Attention Algorithm](https://raw.githubusercontent.com/aadishagr/hokagehub.github.io/refs/heads/main/content/docs/assets/images/ringatten/ring_attention_algo.png#center)

Ring attention leverages the concept of a ring structure, where hosts are arranged sequentially: host-1, host-2, …, host-N. During blockwise attention and feedforward computations, each host efficiently coordinates by simultaneously sending key-value blocks to the next host and receiving key-value blocks from the previous host. This overlapping of block transfers with computation ensures seamless coordination. Specifically, for any host-i, while computing attention between its query block and a key-value block, it concurrently sends key-value blocks to host-(i + 1) and receives key-value blocks from host-(i - 1). If the computation time surpasses the transfer time, there is no additional communication cost. This overlapping mechanism is applicable to both forward and backward passes, utilizing the same operations and techniques.
![Diagram 13: Ring attention flow](https://raw.githubusercontent.com/aadishagr/hokagehub.github.io/refs/heads/main/content/docs/assets/images/ringatten/ring_attention_flow.png#center)

The diagram (a) illustrates a ring of hosts, where each host holds a query block. Key-value blocks circulate through this ring for attention and feedforward computations, processed block-by-block. During attention computation, each host sends key-value blocks to the next host and receives key-value blocks from the previous host. This communication is seamlessly overlapped with the blockwise attention and feedforward computations.
Diagram (b) illustrates the computation of the original Transformer block-by-block. Each host handles one iteration of the query's outer loop, while key-value blocks rotate among the hosts. As shown, a device begins with the first query block on the left and iterates over the horizontally positioned key-value blocks. The query block, combined with the key-value blocks, is used to compute self-attention, and the output is then passed to the feedforward network.

### **Minimum block size and memory requirement**
The block size is denoted as c and the hidden size as d. When computing blockwise self-attention, we need 2dc² FLOPs to calculate attention scores using queries and keys, and another 2dc² FLOPs to multiply these attention scores by values. In total, this amounts to 4dc² FLOPs. For communication, both key and value blocks require 2cd bytes each, resulting in a combined communication demand of 4cd bytes. To overlap communication with computation, the condition 4dc²/F≥4cd/B must be met. This implies that the block size cc should be at least F/B, meaning the block size needs to be larger than the ratio of FLOPs to bandwidth.
A host needs to store multiple blocks: one block size for the current query block, two block sizes for the current key and value blocks, and two block sizes for receiving key and value blocks. Additionally, storing the output of blockwise attention and feedforward requires one block size, as the output retains the shape of the query block. In total, six blocks are needed, equating to 6bch bytes of memory. Notably, the blockwise feedforward network has a maximum activation size of 2bch. Therefore, the total maximum activation size remains 6bch bytes. This demonstrates the advantage of linear memory scaling with respect to the block size c, independent of the input sequence length s.

_If you’re interested in learning more about Transformers or exploring how they can be applied to your projects, feel free to reach out in the comments below!_

