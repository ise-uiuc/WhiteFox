
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 7, 1)
    def forward(self, _input):
        v1 = self.conv(_input)
        return v1
# Inputs to the model
_input = torch.randn(1, 2, 4, 32)
