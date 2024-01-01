
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 2, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Identity(2, 2, 2)
    def forward(self, x):
        t1 = self.conv1(x)
        t2 = self.conv2(t1)
        t3 = torch.dropout(t2, 0.2)
        t4 = self.conv1(x)
        t5 = self.conv2(t4)
        t6 = torch.dropout(t5, 0.5)
        return t3, t6
# Inputs to the model
x = torch.randn(3, 1, 32, 32)
