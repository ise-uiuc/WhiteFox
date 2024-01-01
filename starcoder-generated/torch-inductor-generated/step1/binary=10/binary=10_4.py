
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 512, bias=False)
    
    def forward(self, x):
        p1 = self.linear(x)
        p2 = x + p1
        return p2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(16, 128)
p1 = torch.randn(16, 512)
