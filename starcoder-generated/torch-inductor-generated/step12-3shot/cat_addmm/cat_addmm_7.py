
class Model(torch.nn.Module):
    def __init__(self, N=int):
        super().__init__()
        self.fc1 = torch.nn.Linear(4 * N, 32 * N)
        self.fc2 = torch.nn.Linear(32 * N, 16 * N)

    def forward(self, x1):
        v1 = self.fc1(x1)
        v2 = self.fc2(v1)
        return v2

# Initializing the model
m = Model(512)

# Inputs to the model
x1 = torch.randn(1, 4 * 512)
