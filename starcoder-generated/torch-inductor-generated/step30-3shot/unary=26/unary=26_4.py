
t4 = torch.nn.ConvTranspose2d(2, 2, 2, stride=2)
t5 = t4(x)
negative_slope = 0.13
t1 = t4(x)
t2 = t1 > 0
t3 = t1 * negative_slope
t4 = torch.where(t2, t1, t3)
return (t4 - 0.5) * 6
# Inputs to the model
x = torch.randn(3, 2, 10, 10)
