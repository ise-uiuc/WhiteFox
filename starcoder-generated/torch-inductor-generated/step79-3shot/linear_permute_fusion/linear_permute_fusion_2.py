
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x2):
        v0 = x2
        for k1 in range(v0.size(1)):
            v1 = torch.nn.functional.linear(v0, self.linear.weight, self.linear.bias)
            v2 = v1.permute(0, 2, 1)
            v0[:, k1, :, :] = v2
        return v0


# Inputs to the model
x0 = torch.randn(1, 2, 2)
x1 = x0 + 1.0
