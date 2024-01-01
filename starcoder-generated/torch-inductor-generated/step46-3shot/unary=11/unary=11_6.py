
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.__0 = torch.nn.Conv2d(3, 11, 4, stride=2)
        self.__1 = torch.nn.ReLU()
        self.__3 = torch.nn.ConvTranspose2d(11, 9, 3, output_padding=0)
        self.conv_transpose = torch.nn.ConvTranspose2d(9, 7, 3, output_padding=1)
    def forward(self, x1):
        v1 = self.__0(x1)
        v2 = self.__1(v1)
        v4 = self.__3(v2)
        v5 = v4 + 3
        v6 = torch.clamp_min(v5, 0)
        v7 = torch.clamp_max(v6, 6)
        v8 = v7 / 6
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
