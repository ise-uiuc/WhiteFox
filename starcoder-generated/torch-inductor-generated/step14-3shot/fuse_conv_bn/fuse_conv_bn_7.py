
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 4)
        self.conv2 = torch.nn.Conv2d(1, 1, 2)
        self.conv3 = torch.nn.Conv2d(1, 1, 1)
    def forward(self, x):
        # TODO
        t1 = self.conv1(x)
        t2 = self.conv2(t1) if t1 is not None else None
        t7 = self.conv3(t2) if t2 is not None else None
        return t7
# Inputs to the model
x = torch.randn(1, 3, 3, 3)
