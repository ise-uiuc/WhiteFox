
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 5)
        self.linear2 = torch.nn.Linear(5, 3)
        self.linear3 = torch.nn.Linear(8, 9)
        self.linear4 = torch.nn.Linear(9, 4)
    def forward(self, x1):
        v1 = torch.cos(x1)
        v2 = torch.nn.functional.linear(v1, self.linear4.weight, self.linear4.bias)
        v3 = v2.permute(0, 2, 1)
        v4 = torch.nn.functional.linear(v3, self.linear3.weight, self.linear3.bias)
        v5 = torch.sigmoid(v4)
        v6 = v1.view(1, 480)
        v7 = torch.sigmoid(torch.nn.functional.linear(v6, self.linear1.weight, self.linear1.bias))
        v8 = torch.max(v5, dim=-1, keepdim=False)[0]
        v9 = torch.nn.functional.linear(v8, self.linear2.weight, self.linear2.bias)
        return v9
# Inputs to the model
x1 = torch.randn(6, 480)
