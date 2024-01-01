
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.flatten = torch.nn.Flatten(0, 1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v4 = self.sigmoid(v3)
        x2 = v4.expand_as(v1)
        v3 = v3.unsqueeze(1)
        v3 = v3.to(v1.dtype)
        v4 = (v1 * v3) / x2
        v4 = v4.narrow(dim=1, start=0, length=1)
        v5 = v4.flatten(2)
        v5 = torch.stack(v5, dim=1)
        return v5
# Inputs to the model
x1 = torch.randn(1, 2, 2)
