
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(10, 1)
 
    def forward(self, x):
        y = self.linear(x)
        y_hat = torch.sigmoid(y)
        return y_hat

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
