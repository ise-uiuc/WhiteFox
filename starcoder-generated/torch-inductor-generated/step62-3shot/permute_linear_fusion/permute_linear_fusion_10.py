
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 1)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = torch.nn.functional.relu(v1)
        v3 = torch.nn.functional.linear(v3, torch.nn.ReLU6(self.linear2.weight), torch.nn.ReLU6(self.linear2.bias))
        v3 = torch.nn.functional.hardtanh(v3, -1.0, 1.0)
        if_then = torch.sum(v1 + v3)
        if if_then == 0:
            return v2.permute(0, 2, 1)
        v2 = v2.squeeze(dim=0)
        v3 = v3.permute(1, 0)
        v4 = torch.nn.ReLU6()(torch.nn.functional.hardtanh(torch.matmul(v2, v3), -1.0, 1.0))
        return torch.nn.functional.tanh(v2 + v4)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
