
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.reshape = torch.nn.Flatten()
        self.linear = torch.nn.Linear(3072, 10)
    def forward(self, x1):
        v1 = self.reshape(x1)
        v2 = self.linear(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
