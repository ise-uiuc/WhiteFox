
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(60, 80)
 
    def forward(self, x):
       y = self.linear(x)
       y = y + x
       return y


# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(20, 100)
