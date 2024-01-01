
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.Conv2d(28, 16, kernel_size=1, stride=1, padding=0)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(28, 6, kernel_size=3, stride=2, padding=1)
        self.relu = torch.nn.ReLU()
        self.relu_inplace = torch.nn.ReLU(inplace=True)
    def forward(self, input):
        v0 = self.conv_transpose1(input)
        v1 = self.conv_transpose2(v0)
        v2 = self.relu(v1)
        v3 = self.relu_inplace(v1)
        return v0, v2, v3
# Inputs to the model
input = torch.randn(10, 28, 35, 35)
