
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        v = torch.bmm(x.unsqueeze(0), torch.ones(2, 2, 2))
        return v

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(2, 2, 1)
