
# tokenization

converts between strings and sequences of integers (tokens)

byte-pair encoding BPE

# architecture

original transformer model

# implementation

- bpe
- transformer, cross entropy loss, adamW optimizer, training loop
- train on tinystories and openwebtext

# gpu tricks

the memory is placed at warehouse
the compute is working at the factory
the transfer of memory data from warehouse to the factory is the bandwidth -> the bottleneck

- trick to organize computation to maximize utilization of GPUs by minimizing data movement

# inference

prefill and decode
prefill -tokens are given can process all at once (compute-bound)
decode - need to generate one token at a time (memory-bound)