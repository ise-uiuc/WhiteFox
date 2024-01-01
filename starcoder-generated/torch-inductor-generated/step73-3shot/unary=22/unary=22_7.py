
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128 * 128, 1)
 
    def forward(self, x1):
        v1 = self.linear(x1.view(x1.shape[0], -1))
        v2 = torch.tanh(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
