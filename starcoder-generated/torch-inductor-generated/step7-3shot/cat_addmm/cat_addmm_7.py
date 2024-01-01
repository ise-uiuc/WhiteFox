
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1):
        x1 = x1.view(-1, 3, 32, 32)
        x1 = torch.nn.functional.max_pool2d(x1, [2, 2], [2, 2])
        x1 = x1.view(-1, 256)
        x1 = self.my_func(x1)
        v1 = x1.unsqueeze(2)
        v1 = v1.unsqueeze(3)
        v2 = v1.expand(-1, -1, 21, 21)
        v3 = x1.unsqueeze(2)
        v3 = v3.unsqueeze(3)
        v4 = v3.expand(-1, -1, 21, 21)
        v5 = torch.cat([v2, v4], 1)
        return v5

    def my_func(self, x1):
        return torch.cat([x1], dim=0)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 96, 40, 40, dtype=torch.float64)
