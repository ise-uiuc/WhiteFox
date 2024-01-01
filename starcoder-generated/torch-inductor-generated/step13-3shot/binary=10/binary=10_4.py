
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 1)
        self.other = torch.from_numpy(np.array([[0.5973, -0.2148, -0.2483, 0.9334, -0.7852]])

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5)
