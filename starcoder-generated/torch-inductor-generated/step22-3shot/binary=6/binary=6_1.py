
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(__SIZE_1__, __SIZE_2__)

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - __VALUE___
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(__DIM_1__)
