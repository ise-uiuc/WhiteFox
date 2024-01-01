
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(13, 26, bias=False)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - 13
        v3 = torch.nn.ReLU()(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 13)
