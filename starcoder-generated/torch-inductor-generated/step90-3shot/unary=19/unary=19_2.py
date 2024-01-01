
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4*8, 1)
 
    def forward(self, x1):
        y = self.linear(x1)
        y = torch.sigmoid(y)
        return y


# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4*8)
