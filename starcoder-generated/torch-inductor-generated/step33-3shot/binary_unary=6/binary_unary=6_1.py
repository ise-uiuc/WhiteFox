
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.l1 = nn.Linear(5, 2, 1)
        self.l2 = nn.Linear(5, 1, 1)
 
    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 0) 
        x = self.l1(x) - x1
        x0 = self.l2(x)
        return x0
 
class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.l1 = nn.Linear(5, 2, 1)
        self.l2 = nn.Linear(5, 1, 1)
 
    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 0) 
        x = self.l1(x) - x2
        x0 = self.l2(x) 
        return x0

# Initializing the model
m1 = Net1()
m2 = Net2()

# Inputs to the model
x1 = torch.randn(1, 5)
x2 = torch.randn(1, 5)

# Output from the model
__output1__ = m1(x1, x2)
__output2__ = m2(x1, x2)

