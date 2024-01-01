
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 3)
        self.conv = torch.nn.Conv2d(1, 3, 1, 1)
        self.linear2 = torch.nn.Linear(3, 4)
    def forward(self, x):
        y = self.conv(x)
        x1 = y.permute(-1, 0, 1, 2)
        return torch.nn.functional.linear(x1, self.linear1.weight, self.linear1.bias) + self.conv(x) + torch.nn.functional.relu(torch.nn.functional.linear(x1, self.linear1.weight, self.linear1.bias)) + self.linear2(x1)
# Inputs to the model
x = torch.randn(1, 1, 2, 2)
