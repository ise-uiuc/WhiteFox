
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x2 = torch.nn.functional.softplus(v2)
        x3 = torch.nn.functional.sigmoid(x2)
        x4 = x2.squeeze(dim=0)
        return x4
# Inputs to the model
x1 = torch.randn(1, 2, 1)
