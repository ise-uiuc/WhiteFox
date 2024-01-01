
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 1, stride=1, padding=0)
    def forward(self, input):
        conv = self.conv(input)
        clamp_max = torch.clamp(conv, max=0.015)
        return clamp_max
# Inputs to the model
input = torch.rand(1, 3, 127, 127)
