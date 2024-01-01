
class Model(torch.nn.Module):
    # Please provide the definition of the constructor of the model
    # Constructor's definition begins
    def __init__(self):
    # Ends here
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1):
        y1 = self.linear(x1)
        y2 = torch.sigmoid(y1)
        y3 = y1 * y2
        return y3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
