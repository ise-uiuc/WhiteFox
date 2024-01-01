
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.sigmoid(torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias))
        v3 = v1 * self.sigmoid(v2)
        v4 = torch.max(v3, dim=-1)[1]
        v5 = v4 * v1
        v6 = v1.permute(2, 0, 1) + v5
        v7 = v5 * v6
        return v7
# Inputs to the model
x1 = torch.randn(1, 2, 2)
