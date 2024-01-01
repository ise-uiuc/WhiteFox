
with torch.no_grad():
    s1 = torch.randn(input_dim, hidden_dim).tril_(-27)
    s1 = torch.mm(s1, s1)
    for _ in range(num_block):
      ...
return s1

# Inputs to the model
x1 = torch.randn(hidden_dim, hidden_dim, hidden_dim)
y1 = torch.randn(hidden_dim, hidden_dim, hidden_dim)
