
q = torch.randn(16, 28, 32)
k = torch.randn(16, 32, 48)
v = torch.randn(16, 32, 48)
dropout_p = 0.5
scale_factor = 1.0 / math.sqrt(32)
inv_scale_factor = 1.0 / (scale_factor * dropout_p)
m = Model(q, k, v)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
