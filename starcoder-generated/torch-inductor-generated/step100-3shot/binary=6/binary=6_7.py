
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(100, 100, bias=True)
 
    def forward(self, x1_1):
        v1_1 = self.linear(x1_1)
        v2_1 = v1_1 - 1.7022768570275715e+308
        return v2_1

# Initializing the model
m = Model()

# Inputs to the model
x1_1 = torch.randn(1, 100)
