
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.softmax = torch.nn.Softmax(dim=-1)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x2 = torch.nn.functional.relu(v2)
        v3 = torch.max(x2, dim=1)[-1]
        v3 = v3.unsqueeze(-1)
        v4 = v3 * x1
        v5 = torch.mean(v4, dim=1, keepdim=True)
        v6 = torch.cat((v5, v5), dim=1)
        v7 = self.softmax(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 2, 2)
