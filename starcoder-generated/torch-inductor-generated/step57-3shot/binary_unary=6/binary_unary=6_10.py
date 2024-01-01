
class Model(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linear = torch.nn.Linear(size, size, bias=True)
 
    def forward(self, x):
        out = self.linear(x)
        out = out - 3
        out = torch.relu(out)
        return out

# Initializing the model
m = Model(128)

# Inputs to the model
x = torch.randn(1, 128)
