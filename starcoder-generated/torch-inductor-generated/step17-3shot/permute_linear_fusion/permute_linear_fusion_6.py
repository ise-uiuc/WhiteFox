
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 10)
        self.linear2 = torch.nn.Linear(10, 5)
    def forward(self, x1):
        v1 = torch.matmul(x1, self.linear.weight)
        v2 = torch.matmul(v1, self.linear2.weight)
        return torch.relu(v2)
# Inputs to the model
x1 = torch.randn(2, 5)
