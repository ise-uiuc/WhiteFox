
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = x1.transpose(1, 2)
        v3 = torch.matmul(v1, v2)
        return v2 + (v3)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2, 2)
