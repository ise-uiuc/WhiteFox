
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(64, 64)
 
    def forward(self, x):
        v1 = self.linear_1(x)
        x_r = v1 + x
        return x_r

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 64)
