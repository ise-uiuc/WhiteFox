
class Model(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.linear = torch.nn.Linear(64 * 64 * 3, 1)
 
    def forward(self, x1):
        t1 = self.linear(x1.view(x1.shape[0], -1))
        return torch.tanh(t1)

# Initializing the model
m = Model(3)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
