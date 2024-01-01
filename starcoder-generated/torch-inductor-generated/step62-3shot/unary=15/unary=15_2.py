
model = torch.nn.MultiheadAttention(embed_dim=80, num_heads=8, dropout=0.1)
# Inputs to the model
key = torch.randn(query.size(0), query.size(1), embed_dim)
value = torch.randn(key.size(0), value.size(1), embed_dim)
