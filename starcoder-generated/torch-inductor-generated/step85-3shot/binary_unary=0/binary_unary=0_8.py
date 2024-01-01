
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 5, stride=1, padding=2, bias=False, padding_mode='zeros')
        self.conv2 = torch.nn.Conv2d(16, 16, 5, stride=1, padding=2, bias=False)
    def forward(self, x1, x2):
        v1 = torch.mul(x1, x2)
        v2 = self.conv1(v1)
        v3 = torch.nn.functional.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 + v2
        v6 = torch.nn.functional.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randint(0, 10, (1, 16, 64, 64))
x2 = torch.randint(0, 10, (1, 16, 64, 64))
