
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 32)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = torch.tanh(v1)
        return v2

# Initializing the model
m = Model()

Input to the model
x1 = torch.randn(1, 8)
