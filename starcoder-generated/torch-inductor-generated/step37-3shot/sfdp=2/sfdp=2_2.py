
m = torch.nn.Transformer(d_model=1024, num_encoder_layers=1)

# Inputs to the model
x1 = torch.randn(4, 1024, 3)
x2 = torch.randn(4, 1024, 8)
x3 = torch.randint(1024, (4, 1024, 1,), dtype=torch.long)
