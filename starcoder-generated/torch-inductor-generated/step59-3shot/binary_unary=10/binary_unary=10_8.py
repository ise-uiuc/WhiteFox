
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(3 * 32 * 32, 512)

    def forward(self, x1):
        v1 = self.fc1(x1)
        v2 = v1 + torch.rand_like(v1)
        v3 = torch.nn.ReLU()(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3 * 32 * 32)
