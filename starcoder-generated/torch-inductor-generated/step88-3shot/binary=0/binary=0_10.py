
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(15, 10, 11, stride=1, padding=7)
    def forward(self, input_tensor, other=None):
        v1 = self.conv(input_tensor)
        if other == None:
            other = torch.randn(v1.shape)
        v2 = v1 + other
        return v2
# Inputs to the model
input_tensor = torch.randn(1, 15, 161, 161)
