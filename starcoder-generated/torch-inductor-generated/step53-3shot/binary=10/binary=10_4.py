
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(20, 3)

    def forward(self, x1, other=torch.rand(3, 20)):
        v1 = self.layer1(x1)
        v2 = v1 + other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.rand(1, 20)
