
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        x2 = torch.nn.functional.relu(v1)
        v2 = torch.nn.functional.linear(x2, self.linear.weight, self.linear.bias)
        v2 = torch.sigmoid(v2)
        v3 = torch.max(v2, dim=-1)[0].sum()
        v3 = v3 + v3
        v3 = torch.max(v2, dim=-1)[0]
        v3 = v3 + torch.max(v2, dim=-1)[1]
        x3 = v3 + x2
        v3 = x1.view(1, 4)
        y = x3 - v3
        return y
# Inputs to the model
x1 = torch.randn(1, 2, 2)
