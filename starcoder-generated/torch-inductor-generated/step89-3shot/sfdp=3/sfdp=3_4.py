
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul_1 = torch.nn.Linear(2048, 2048)

    def forward(self, x1):
        v1 = self.matmul_1(x1)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2048)
