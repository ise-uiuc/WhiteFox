
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(128, 64)
        self.linear2 = torch.nn.Linear(64, 1)
        self.activation = torch.nn.LeakyReLU(0.2)
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = self.activation(v5)
        v7 = v2 * v6
        v8 = self.linear2(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 128)
