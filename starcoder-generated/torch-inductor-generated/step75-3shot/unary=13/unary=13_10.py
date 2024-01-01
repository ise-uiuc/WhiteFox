
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(512, 2048, bias=True)

    def forward(self, v1):
        h1 = self.linear(v1)
        h2 = h1.sigmoid()
        output = h1 * h2
        return output

# Initializing the model
m = Model()

# Inputs to the model
v1 = torch.randn(1, 512)
