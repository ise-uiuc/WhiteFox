
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        block_0 = [torch.nn.Conv2d(64, 32, 3, 1, 0, bias=False), torch.nn.BatchNorm2d(32)]
        block_1 = [torch.nn.ReLU()]
        block_2 = [Block(), torch.nn.BatchNorm2d(32)]
        block_3 = [torch.nn.ReLU()]
        block_4 = [Block()]
        block_5 = [torch.nn.Conv2d(128, 32, 3, 1, 0, bias=False), torch.nn.ReLU()]
        block_6 = [Block()]
        block_7 = [torch.nn.Conv2d(32, 3, 3, 1, 0, bias=False), torch.nn.ReLU()]
        self.features = torch.nn.Sequential(*block_0, *block_1, *block_2, *block_3, *block_4, *block_5, *block_6, *block_7)
        self.x = torch.nn.Sequential()
        self.x[0].bias.data.fill_(5)
    def forward(self, v1):
        result = (v1, )
        result += (self.features[0](v1) + self.features[1](v1).repeat(1, 1, 1, 1), )
        result += (self.features[2](v1) + self.features[3](v1), )
        result += (self.features[4](result[3]) + self.features[5](result[3]), )
        return result[4]
# Inputs to the model
x1 = torch.randn(1, 64, 32, 32)
