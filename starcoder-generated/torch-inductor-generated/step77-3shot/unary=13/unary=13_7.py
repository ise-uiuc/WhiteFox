
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(2, 3)
 
    def forward(self, x1):
        y1 = self.linear1(x1)
        y2 = torch.sigmoid(y1)
        y3 = y1 * y2
        return y3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2)
