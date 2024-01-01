
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Conv2d(3, 128, 1, stride=1)
        self.linear2 = torch.nn.Conv2d(3, 64, 1, stride=1)
        self.linear3 = torch.nn.Conv2d(3, 16, 1, stride=1)
 
    def forward(self, x, other):
        v2 = self.linear2(x) + other
        v3 = torch.nn.functional.relu(v2)
        v4 = self.linear3(x) + other
        v5 = torch.nn.functional.relu(v4)
        v6 = self.linear1(x) + other
        v7 = torch.nn.functional.relu(v6)
        return v7

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
other = torch.randn(1, 3, 64, 64)
