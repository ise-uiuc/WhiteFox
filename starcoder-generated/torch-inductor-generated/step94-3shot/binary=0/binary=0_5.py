
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(10, 25, 1, stride=1, padding=1)
    def forward(self, input_tensor=None, padding1=None):
        v1 = self.conv(input_tensor)
        if padding1 == None:
            padding1 = torch.randn(v1.shape)
        v2 = v1 - padding1
        return v2
# Inputs to the model
input_tensor = torch.randn(1, 10, 64, 64)
