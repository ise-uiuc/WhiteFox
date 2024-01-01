
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.Tensor([0.5]))

    def forward(self, x1):
        v1 = torch.matmul(x1, self.weight.T)
        v2 = v1 > 0
        v3 = v1 * self.weight[0]
        v4 = torch.where(v2, v1, v3)
        return v4


# Initializing the model
m = m = Model(0.1)

# Inputs to the model
x1 = torch.randn(2, 8)
