
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 6)
 
    def forward(self, x1, x2):
        v4 = self.linear(torch.cat((x1, x2), dim=1))
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
x2 = torch.randn(1, 8)
