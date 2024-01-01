
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.W_0 = torch.nn.Linear(10, 20, bias=False)
 
    def forward(self, x):
        v1 = self.W_0(x)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(3, 10)
