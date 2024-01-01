 with other being a keyword argument
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1, other=torch.randn(5, 5)):
        v1 = self.linear(x1)
        v2 = v1 + other
        v3 = torch.nn.ReLU()(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
