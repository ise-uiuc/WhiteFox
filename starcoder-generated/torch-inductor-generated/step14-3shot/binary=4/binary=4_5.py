
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_weights = torch.nn.Parameter(torch.rand(10, 20))
        self.linear_bias = torch.nn.Parameter(torch.normal(0, (20)))

    def forward(self, x):
        v1 = x.permute(0, 2, 3, 1).contiguous().view(-1, 20)
        v2 = v1 + self.linear_bias.view(1, 20)
        v3 = torch.matmul(v2, self.linear_weights.view(20, 10))
        v4 = v3.view(-1, 64, 64, 10).permute(0, 3, 1, 2)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10, 64, 64)
