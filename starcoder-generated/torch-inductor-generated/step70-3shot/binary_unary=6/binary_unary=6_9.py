
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, 1, 1)
        self.conv2 = nn.Conv2d(4, 8, 3, 1, 1)
        self.fc1 = nn.Linear(8 * 28 * 28, 10)
 
    def forward(self, x1):
        o1 = self.conv1(x1)
        o2 = self.conv2(o1)
        o3 = self.fc1(o2.view((-1, 8 * 28 * 28)))
        return o3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
