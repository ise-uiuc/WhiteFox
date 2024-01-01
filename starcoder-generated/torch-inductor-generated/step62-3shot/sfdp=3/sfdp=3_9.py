
new_layer = torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0)

# Inputs to the model
x1 = torch.randn(1, 256, 512)
x2 = torch.randn(1, 512, 256)
