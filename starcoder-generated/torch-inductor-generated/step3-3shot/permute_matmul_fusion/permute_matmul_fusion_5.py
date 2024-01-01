
x = torch.randn(1,2,2)
a = x[0:1].permute(0,2,1)
b = x[0:1]
y = torch.matmul(b, a)
print(x)
# Inputs to the model
x = torch.randn(1,2,2)
