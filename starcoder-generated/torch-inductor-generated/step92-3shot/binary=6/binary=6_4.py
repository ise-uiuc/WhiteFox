
t = torch.nn.Linear(8, 16)

# Input to the model
x1 = torch.randn(10, 8)
y1 = torch.randn(10)

v1 = t(x1)
v2 = v1 - y1
return v2

