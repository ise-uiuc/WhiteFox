
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(768, 768)

    def forward(self, x):
        x = self.linear(x)
        x = F.relu(x)
        return x

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(32, 4, 768)
