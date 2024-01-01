
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.linear2 = torch.nn.Linear(10, 10)
    def forward(self, x1):
        x2 = self.linear1(x1)
        x3 = self.linear2(x1)
        x4 = x2 - x3
        x5 = F.relu(x4)
        return x5
# Inputs to the model
x1 = torch.randn(2, 10)
