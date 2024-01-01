
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(512, 1024, 3, stride=1, padding=0, bias=False)
        self.relu_1 = torch.nn.ReLU(inplace=True)
    def forward(self, input_1):
        v1 = self.relu_1(self.conv_transpose_1(input_1))
        return v1
# Inputs to the model
input_1 = torch.randn(1, 512, 1, 1)
