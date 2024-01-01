
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=1, out_features=10)
 
    def forward(self, x):
        x = self.linear(x)
        x = torch.tanh(x)
        return x

# Initializing the model
m = Net()

# Inputs to the model
x = torch.randn(1, 1)
