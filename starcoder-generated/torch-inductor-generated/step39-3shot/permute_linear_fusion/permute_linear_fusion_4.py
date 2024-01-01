
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x2 = torch.nn.functional.relu(v2)
        v2 = x2.detach()
        v2 = torch.mean(v2, dim=1)
        v4 = torch.tensor([-1.0])
        v4 = torch.nn.functional.linear(v2, torch.transpose(v2, 0, 1), v4)
        v3 = -v4 + 1.0
        return x2 - v3.to(x2.dtype)
# Inputs to the model
x1 = torch.randn(2, 2)
