
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)

    def forward(self, x1):
        return self.linear(x1) + other

# Initializing the model
m = Model(other=torch.randn(8, 3))
# Input to the model
x1 = torch.randn(1, 3)
