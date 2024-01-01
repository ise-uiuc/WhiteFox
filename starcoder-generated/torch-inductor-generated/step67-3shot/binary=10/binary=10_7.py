
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 20, bias=None)
 
    def forward(self, x):
        y = self.linear(x)
        # y is a random initialized tensor
        # now we manually set part of it to be 1,
        # because the bias value of self.linear
        # has not been initialized here
        y[:, [0, 1, 2, 3]] = 1
        return y

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 10)
