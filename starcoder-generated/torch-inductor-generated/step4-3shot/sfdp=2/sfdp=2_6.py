
Attention_Mechanism = torch.nn.MultiheadAttention(embed_dim=EMBEDDING_SIZE, 
                                                 num_heads=NUM_HEADS, dropout=DROPOUT, bias=False)

# Initializing the model
m = Attention_Mechanism

# Inputs to the model
x1 = torch.randn(BATCH_SIZE, LEN_Q, EMBEDDING_SIZE)
x2 = torch.randn(BATCH_SIZE, LEN_K, EMBEDDING_SIZE)
x3 = torch.randn(BATCH_SIZE, LEN_V, EMBEDDING_SIZE)
