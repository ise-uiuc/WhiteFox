
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)

    def forward(self, x2):
        v2 = self.linear(x2)
        v6 = torch.sigmoid(v2)
        return v6

# Initializing the model
m = Model()

# Input to the model
x2 = torch.randn(1,8)
