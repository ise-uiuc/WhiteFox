
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 3, 3)
        self.conv2 = torch.nn.Conv1d(1, 3, 3)
    def forward(self, x):
        a1 = self.conv1(x)
        a2 = self.conv2(x)
        b1 = a1
        b2 = torch.nn.functional.relu(a1)
        z1 = torch.nn.functional.dropout(b1, p=0.1)
        z2 = torch.nn.functional.dropout(b2, p=0.1)
        o1 = z1 + z2
        return o1
# Inputs to the model
x1 = torch.randn(1, 1, 23, 22)
