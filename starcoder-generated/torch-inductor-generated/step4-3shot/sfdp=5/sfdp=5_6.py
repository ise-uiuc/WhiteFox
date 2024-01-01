
t1 = query @ key.transpose(-2, -1) 
t1 = t1 / math.sqrt(query.size(-1))
t2 = t1 + attn_mask
t3 = torch.softmax(t2, dim=-1)
t4 = torch.dropout(t3, dropout_p, True)
output = t4 @ value

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 5, 128)
key = torch.randn(1, 5, 128)
value = torch.randn(1, 5, 128)
attn_mask = torch.randn(1, 5)
dropout_p = 0.1
