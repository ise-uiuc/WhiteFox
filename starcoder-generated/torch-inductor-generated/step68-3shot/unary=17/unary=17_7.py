
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.squeeze(v1, dim=1)
        x2 = torch.tanh(v2)
        return x2
# Inputs to the model
x1 = torch.randn(1, 10)
x2 = torch.randn(1, 20)
