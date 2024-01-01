
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
    def forward(self, x1):
        v0 = torch.unsqueeze(x1, 0)
        v1 = torch.squeeze(v0)
        v2 = torch.squeeze(v1)
        v3 = x1.permute(0, 2, 1)
        v4 = torch.nn.functional.linear(v3, self.linear.weight, self.linear.bias)
        return v4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
