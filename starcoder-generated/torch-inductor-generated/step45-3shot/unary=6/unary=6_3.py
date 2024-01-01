
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 20, 5, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(20, 50, 5, stride=1, padding=0)
        self.fc1 = torch.nn.Linear(800, 500)
        self.fc2 = torch.nn.Linear(500, 10)
    def forward(self, x1):
        t1 = self.conv1(x1)
        t2 = self.conv2(t1)
        t3 = t2.view(-1, 800)
        t4 = torch.nn.Tanh(self.fc1(t3))
        t5 = torch.nn.ReLU6(self.fc2(t4))
        t6 = t5 / 6
        return t6.unsqueeze(-1)

# Inputs to the model
x1 = torch.randn(1, 1, 28, 28)
