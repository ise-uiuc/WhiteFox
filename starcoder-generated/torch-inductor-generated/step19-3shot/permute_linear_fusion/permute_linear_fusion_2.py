
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.ReLU = torch.nn.ReLU()
    def forward(self, x1):
        return torch.matmul(self.ReLU(x1), self.linear.weight)
# Inputs to the model
x1 = torch.randn(1, 1, 2)
