
class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(784, 128)
        self.fc2 = torch.nn.Linear(128, 62)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x - 0.5
        return x

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(3, 1, 28, 28)
