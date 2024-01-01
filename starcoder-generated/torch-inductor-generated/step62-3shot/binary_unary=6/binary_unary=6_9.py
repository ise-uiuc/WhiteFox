
class Model_add(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(3, 6)

    def forward(self, x1):
        y0 = self.linear_1(x1)
        y1 = y0 - 2.0
        y2 = F.relu(y1)
        return y2

# Initializing the model
m = Model_add()

# Inputs to the model
x1 = torch.randn(1, 3)
