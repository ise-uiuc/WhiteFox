
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(6, 5)
 
    def forward(self, x1):
        v1 = self.linear(x1[:, :, 0, 0])
        v2 = torch.stack([v1, v1 + 1, v1 + 2, v1 + 3, v1 + 4])
        v3 = torch.max(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 6, 16, 16)
