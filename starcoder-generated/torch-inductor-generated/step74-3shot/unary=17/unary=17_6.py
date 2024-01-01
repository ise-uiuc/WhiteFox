
class Model(torch.nn.Sequential):
    def __init__(self):
        super().__init__()
        self.add_module('conv_transpose', torch.nn.ConvTranspose2d(1936, 9216, kernel_size=(14, 14), stride=(1, 1)))
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.relu(v1)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1936, 14, 14)
