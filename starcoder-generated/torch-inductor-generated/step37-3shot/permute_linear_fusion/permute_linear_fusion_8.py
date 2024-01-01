
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x2 = torch.nn.functional.relu(v2)
        v3 = x2.detach()
        v4 = (v3 == -1).to(v3.dtype)
        v5 = v4.to(v3.dtype)
        x3 = v3.permute(0, 2, 1)
        x3 = torch.addcmul(x3, v5, (v3 == -1).to(v3.dtype))
        return torch.nn.functional.linear(x3, self.linear.weight, self.linear.bias)
# Inputs to the model
x0 = torch.randn(1, 1, 2)
