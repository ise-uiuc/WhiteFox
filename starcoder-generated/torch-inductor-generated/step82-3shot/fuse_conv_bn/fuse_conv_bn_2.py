
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.linear2 = torch.nn.Linear(10, 10)
    def forward(self, x1):
        x1 = self.linear1(x1)
        x1 = x1.view(1, 10)
        x1 = self.linear2(x1)
        return x1
# Inputs to the model
x1 = torch.randn(1, 10)
