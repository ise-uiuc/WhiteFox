
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 6, 1, stride=1, padding=1)
    def forward(self, input):
        v1 = self.conv(input)
        v2 = v1 - torch.Tensor([[[
        [1.0, 0.0, 1.0]],

        [[1.0, -1.0, 1.0]],

        [[1.0, 0.0, -1.0]]]]])
        return v2
# Inputs to the model
input = torch.randn(1, 3)
