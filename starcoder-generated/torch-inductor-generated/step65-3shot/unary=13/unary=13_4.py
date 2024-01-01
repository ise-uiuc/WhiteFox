
class Sequential(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(3, 16)
        self.layer2 = torch.nn.Sigmoid()
 
    def forward(self, x1):
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = x2 * x3
        return x4

# Initializing the model
m = Sequential()

# Inputs to the model
x1 = torch.randn(1, 3)
