
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a0 = torch.nn.MaxPool1d(2, 2)
        self.linear_1 = torch.nn.Linear(2, 2)
        self.a1 = torch.nn.MaxPool1d(2, 2)
        self.a2 = torch.nn.AvgPool1d(2, 2)
        self.a3 = torch.nn.MaxPool1d(2, 2)

    def forward(self, x):
        x = self.a0(x)
        x1 = torch.nn.functional.linear(x, self.linear_1.weight, self.linear_1.bias)
        x1 = x1.permute(0, 2, 1)
        x2 = self.a1(x)
        x2 = self.a2(x)
        return self.a3(x)
# Inputs to the model
x = torch.randn(2, 3, 3)
