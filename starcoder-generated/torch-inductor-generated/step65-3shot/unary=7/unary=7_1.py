
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(896, 2048)

    def forward(self, l1, min1, max1, mul1):
        v1 = self.linear(l1)
        v2 = v1 * torch.clamp(min=min1, max=max1, v1 + mul1)
        v3 = v2 / mul1
        return v3

# Initializing the model
m = Model()

# Inputs to the model
l1 = torch.randn(1, 896)
min1 = 0
max1 = 6
mul1 = 6
