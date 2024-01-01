
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 3, stride=1, padding=1)
    def forward(self, input1=None):
        v1 = self.conv(input1)
        return v1
# Inputs to the model
