
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=0)
        self.linear = torch.nn.Linear(64, 10)
 
    def forward(self, x1):
        v1 = torch.sigmoid(self.linear(self.conv2(self.conv1(x1)).view(x1.size(0), -1)))
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
