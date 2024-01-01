
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.sigmoid = torch.nn.Identity()
    def forward(self, x1, h):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = self.sigmoid(v2)
        v4 = v3.squeeze(-1)
        v6 = v4.clone() * h
        return v1
# Inputs to the model
x1 = torch.randn(1, 2, 2)
h = torch.randn(1, 2, 2)
