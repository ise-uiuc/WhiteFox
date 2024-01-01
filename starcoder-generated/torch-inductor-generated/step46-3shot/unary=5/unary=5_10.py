
t1 = conv_transpose(x1)
t2 = t1 * 0.5
t3 = t1 * 0.7071067811865476
t4 = torch.erf(t3)
t5 = t4 + 1
t6 = t2 * t5
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
