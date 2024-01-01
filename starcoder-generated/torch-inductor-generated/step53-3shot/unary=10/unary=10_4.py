
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(20, 64)
 
    def forward(self, x):
        l1 = self.fc(x)
        l2 = l1 + 3
        l3 = torch.clamp_min(l2, 0)
        l4 = torch.clamp_max(l3, 6)
        l5 = l4 / 6
        return l5

# Initializing the model
m = Net()

# Inputs to the model
x1 = torch.randn(1, 20)
