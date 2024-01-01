
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(2*x)


# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(size=(1, 10))

