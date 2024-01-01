
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=0)
        self.linear = torch.nn.Linear(8 * 56 * 56, 10)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v2 = v2.reshape(v2.size()[0], -1)
        v3 = self.linear(v2)
        v4 = F.log_softmax(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
