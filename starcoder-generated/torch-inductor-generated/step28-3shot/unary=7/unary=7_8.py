
class Model(torch.nn.ModuleList):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.linear1 = torch.nn.Linear(64, 32)
        self.linear2 = torch.nn.Linear(32, 16)
        self.linear3 = torch.nn.Linear(16, 4)
 
    def forward(self, x1):
        l1 = self.conv(x1)
        l2 = l1 * torch.clamp(min=0, max=6, self.linear1(x1) + 3)
        l3 = l2 / 6
        l4 = self.linear2(l3)
        l5 = self.linear3(l4)
        return l5

# Initializing the model
m = Model()

# Inputs to the model (random initialization)
x1 = torch.randn(1, 3, 64, 64)
