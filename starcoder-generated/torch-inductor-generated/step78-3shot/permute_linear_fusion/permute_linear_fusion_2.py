
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.softmax = torch.nn.Softmax()
    def forward(self, x1):
        v1 = x1.view(2, 2)
        v2 = x1.view(-1)
        v3 = v2 * v2
        v4 = v3.view(2, -1)
        v5 = torch.nn.functional.relu(v4)
        v6 = v5 - 1.0
        v7 = torch.floor(v6)
        v8 = v7 * v6
        v9 = v8.permute(1, 0)
        v10 = v9 * v1
        v11 = v10.permute(1, 0)
        v12 = torch.nn.functional.tanh(v11)
        v13 = v12 / v11
        v14 = v12.view(-1)
        v15 = self.linear(v14)
        v16 = v15 * v15
        v17 = v16.sum(dim=0)
        v18 = torch.nn.functional.softmax(v17, dim=0)
        return v11, v18
# Inputs to the model
x1 = torch.randn(1, 2, 2)
