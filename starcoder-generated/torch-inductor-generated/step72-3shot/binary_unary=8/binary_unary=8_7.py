
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(35, 83, 1, stride=1, padding=1)
    def forward(self, input):
        t1 = self.conv1(input)
        t2 = torch.relu(t1)
        return t2
# Input to the model
input = torch.randn(1, 35, 128, 256)
