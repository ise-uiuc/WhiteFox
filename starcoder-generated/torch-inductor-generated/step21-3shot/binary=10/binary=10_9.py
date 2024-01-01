
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(12, 10)

    def forward(self, x1):
        return self.linear(x1) + torch.eye(10)

# Initializing the model
m = Model()
x1 = torch.randn(12)
