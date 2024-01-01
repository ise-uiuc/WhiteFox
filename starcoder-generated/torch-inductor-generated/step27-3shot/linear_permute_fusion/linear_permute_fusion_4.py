
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(1, 4)
        self.conv1 = torch.nn.Conv2d(1, 2, 2)
        self.linear2 = torch.nn.Linear(2, 1)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(self.conv1(x1), self.linear1.weight, self.linear1.bias)
        return self.linear2(v1)
# Inputs to the model
x1 = torch.randn(2, 1, 4, 3)
