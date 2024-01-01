
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(1, 1)
        self.linear2 = torch.nn.Linear(1, 1)
    def forward(self, x1):
        x2 = torch.rand(x1.size(0), 2, 1)
        v1 = (x1 + torch.nn.functional.softmax(x1, dim=-1)) * x2
        v2 = x1 * torch.nn.functional.tanh(x2)
        x3 = torch.nn.functional.softmax(x1, dim=-1) + x1 * x2
        x3 = x2 * x3
        x3 = torch.nn.functional.softmax(x2, dim=-1) * x3
        v3 = v1 + x3
        v3 = v3 * v2
        v3 = v3 * v1
        v3 = v2 + v3
        v3 = torch.nn.functional.relu(v3)
        x4 = torch.nn.functional.sigmoid(x2)
        x5 = x2 * x2
        v3 = v1 + x5
        v3 = x4 * v3
        v4 = torch.nn.functional.softmax(x3, dim=-1)
        v4 = x4 + v4
        v4 = x5 * v4
        v4 = x4 * v4
        v4 = v1 ** 2
        v4 = x4 / v4
        v4 = v4 + v2 - v1 + v3
        v4 = v1 - v4
        v4 = v4 * v2
        return torch.nn.functional.sigmoid(v3)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
