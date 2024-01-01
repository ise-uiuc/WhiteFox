
class Model(net):
    def __init__(self,):
        super(Model, self).__init__()
        self.model_partA = nn.Sequential(
            conv_bn_block(3, 16, 64, 2),
            conv_bn_block(16, 32, 32, 2),
            nn.MaxPool2d(3, 2),
            conv_bn_block(32, 64, 16, 2),
            conv_bn_block(64, 128, 8, 2),
            nn.MaxPool2d(2, 2)
        )
        self.upsample2x2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.model_partB = nn.Sequential(
            conv_bn_block(128, 256, 5, 1),
            conv_bn_block(256, 512, 5, 1)
        )

    def forward(self, input_tensor):
        x = self.model_partA(input_tensor)
        x = self.model_partB(x)
        return x
# Inputs to the model
input_tensor = torch.randn(1, 3, 128, 128)
