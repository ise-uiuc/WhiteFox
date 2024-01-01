
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 20)

    def forward(self, x):
        y = self.fc1(x)
        return y + torch.randn(20)

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(2, 10)
