
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = ResBlock(176, 192, 64)
        self.activation = nn.ReLU(inplace=True)
        self.block2 = ResBlock(192, 208, 64)
        self.block3_1 = ResBlock(208, 256, 64, use_shortcut=False)
        self.block3_2 = ResBlock(256, 288, 64)
        self.block3_3 = ResBlock(288, 336, 64)
    def forward(self, x):
        x = self.block1(x)
        x = self.activation(x)
        x = self.block2(x)
        x = self.block3_1(x)
        x = self.activation(x)
        x = self.block3_2(x)
        x = self.block3_3(x)
        return x
# Inputs to the model
x = torch.randn(1, 176, 4, 4)
