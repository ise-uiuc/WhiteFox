
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(100, 200)
 
    def forward(self, x):
        y = self.linear(x)
        y = y * torch.clamp(y + 3, 0, 6) / 6
        return y

# Initializing the model
m = Model()
 
# Inputs to the model
x = torch.randn(128, 100)
