
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
        self.another_tensor = [[-1.7077, -2.7, 0.93703]]

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.another_tensor
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
