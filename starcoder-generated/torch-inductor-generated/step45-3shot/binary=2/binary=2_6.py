
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 40)
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - torch.tensor(0.04553255)
        return v2
# Inputs to the model
x1 = torch.randn(1, 10)
