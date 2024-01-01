
class MyModel(torch.nn.Module):
    def forward(self, a):
        b = torch.cat([a] * 3, 1)
        b = b[:, -((6*a.size(2)**2 - b.size(1))*torch.randint(2, 5) + 1):]
        b = torch.cat([a, b], 1)
        return b

# Initializing the model
m = MyModel()

# Inputs to the model
a = torch.randn(2, 10, 100, 100)
