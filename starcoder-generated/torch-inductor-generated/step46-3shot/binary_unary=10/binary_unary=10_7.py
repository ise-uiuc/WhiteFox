
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1, x2, i1, i2):
        v1 = self.linear(x1)
        v2 = v1 + x2
        v3 = v2[i1, :, i2]
        v4 = torch.relu(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
i1 = torch.tensor([1])
i2 = torch.tensor([2])
