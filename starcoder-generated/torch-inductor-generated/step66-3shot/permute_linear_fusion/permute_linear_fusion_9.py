
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = self.linear(x1)
        t = torch.matmul(x1, v1)
        return torch.matmul(t, v1)
# Inputs to the model
x1 = torch.randn(1, 2)
