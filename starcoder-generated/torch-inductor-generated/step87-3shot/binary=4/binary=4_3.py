
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(128, 512)

    def forward(self, x1, other):
        x2 = x1
        x3 = self.fc1(x2)
        x4 = F.relu(x3)
        x5 = self.fc1(x4)
        x6 = x2 + other
        return x5


# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128)
other = torch.randn(1, 512)
