
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = v1.reshape(v1.shape[0], -1)
        v4 = torch.sum(v2, dim=1, keepdim=False)
        v5 = v3*v4
        return v5
# Inputs to the model
x1 = torch.randn(1, 2, 2)
