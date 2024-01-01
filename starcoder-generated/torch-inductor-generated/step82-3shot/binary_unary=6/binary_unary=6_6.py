
class Model(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linear = torch.nn.Linear(size, size)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - 0.5
        v3 = nn.ReLU()(v2)
        return v3

# Initializing the model
m = Model(128)

# Inputs to the model
x1 = torch.randn(1, 128)
