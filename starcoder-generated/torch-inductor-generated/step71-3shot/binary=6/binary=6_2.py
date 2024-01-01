s
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)
    def forward(self, x):
        v1 = self.linear(x)
        return v1 - x

# Initialize the models
m1 = Model()

