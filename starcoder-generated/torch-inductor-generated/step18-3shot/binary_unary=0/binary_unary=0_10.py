
if condition:
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        def forward(self, x1, x2):
            t1 = self.conv1(x1)
            t2 = t1 + x2
            t3 = torch.relu(t2)
            t4 = self.conv1(t3)
            t5 = t4 + x2
            t6 = torch.relu(t5)
            t7 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)(t6)
            return t7
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = 1
