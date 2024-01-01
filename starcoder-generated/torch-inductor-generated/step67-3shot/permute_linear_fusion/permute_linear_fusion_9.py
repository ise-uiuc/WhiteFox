
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
    def forward(self, x1, x2):
        v0 = x1.clone()
        v1 = v0.squeeze(0)
        v2 = x2.clone()
        v2 = v2.unsqueeze(0)
        v3 = torch.mul(v1, v2)
        v3 = v3.permute(1, 2, 3, 0)
        v4 = torch.transpose(v3, 0, 3)
        v4 = v4.permute(1, 0, 2)
        return torch.nn.functional.linear(v4, self.linear.weight, self.linear.bias)
# Inputs to the model
x1 = torch.randn(1, 3, 3)
x2 = torch.randn(2, 3, 3, 3)
