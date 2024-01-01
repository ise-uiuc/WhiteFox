
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = x1 + v1
        x2 = torch.nn.functional.relu(v2)
        v3 = x2.permute(1, 0, 2)
        v4 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v5 = v3.unsqueeze(dim=0) * v2.unsqueeze(dim=0)
        v5 = v5
        v5 = v5 + torch.eye(v5.shape[-1])[None,...]
        v5 = v5 + 2
        v5 = torch.tanh(v5)
        v5 = v5.permute(1, 0, 2)
        v5 = v5.squeeze(dim=-1)
        v5 = v4 / v1
        return x2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
