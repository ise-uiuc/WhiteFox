
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 2)
        self.linear3 = torch.nn.Linear(2, 2)
        self.linear4 = torch.nn.Linear(2, 2)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear1.weight, self.linear1.bias)
        v3 = torch.nn.functional.linear(v2, self.linear2.weight, self.linear2.bias)
        v4 = torch.nn.functional.linear(v3, self.linear3.weight, self.linear3.bias)
        v5 = torch.nn.functional.linear(v4, self.linear4.weight, self.linear4.bias)
        v6 = self.sigmoid(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 2, 2)
