
m = Model()

# Inputs to the model
query = torch.randn(b, t, n)
key = torch.randn(b, t, n)
value = torch.randn(b, t, n)
