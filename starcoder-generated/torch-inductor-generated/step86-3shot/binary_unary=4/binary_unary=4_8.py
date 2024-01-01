
class Model(torch.nn.Module):
    def __init__(self, other1):
        super().__init__()

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + other1
        v3 = v2
        return v3

# Initializing the model
m = Model(torch.tensor([[0.9]]))
x1 = torch.randn(1, 1, 4)
