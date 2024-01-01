
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x2 = torch.nn.functional.relu(v2)
        v3 = x1.shape
        v4 = v3[0]
        v5 = torch.zeros([v4])
        v5 = v5.to(x1)
        v6 = x2.squeeze(dim=1)
        v7 = torch.nn.functional.sigmoid(v6)
        v4 = x1.squeeze()
        return v7 + v4 + v5
# Inputs to the model
x1 = torch.randn(1, 1, 2, 2)
