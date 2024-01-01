
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 20, 5, 1)
        self.conv2 = torch.nn.Conv2d(20, 50, 5, 1)
        self.fc1 = torch.nn.Linear(4*4*50, 500)
        self.fc2 = torch.nn.Linear(500, 10)
    def forward(self, x1, x2):
        y1 = self.conv1(x1)
        y2 = self.conv2(y1)
        y3 = F.relu(y2)
        y4 = y3.flatten(1)
        y5 = self.fc1(y4)
        y6 = self.fc2(y5)
        y7 = F.log_softmax(y6, dim=1)
        return y7 + x2
# Inputs to the model
x1 = torch.randn(1, 1, 28, 28)
x2 = torch.randn(1)
