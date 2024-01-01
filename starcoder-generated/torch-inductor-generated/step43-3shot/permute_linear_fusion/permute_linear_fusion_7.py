
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.softmax1 = torch.nn.Softmax()
        self.softmax2 = torch.nn.Softmax()
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear1.weight, self.linear1.bias)
        v3 = torch.stack([self.softmax1(v2)[0], self.softmax2(v2)[1]], dim=-1)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
