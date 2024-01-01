
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(512, 512)
 
    def forward(self, x):
        v1 = self.linear(x)
        y = torch.tanh(v1)
        return y

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 512)
