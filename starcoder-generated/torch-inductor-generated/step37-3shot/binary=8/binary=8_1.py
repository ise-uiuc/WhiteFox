 with dynamic shape
class ModelDynamic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, self.dynamic_size, self.dynamic_stride, 1)
        self.conv2 = torch.nn.Conv2d(3, 8, self.dynamic_size, self.dynamic_stride, self.dynamic_padding)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = v1 + v2
        return v3
    @staticmethod
    def get_dynamic_size():
        return random.choice((1,2,3,4,8,16))  # random kernel size
    @staticmethod
    def get_dynamic_stride():
        return random.choice((1, 2, 4, 8))  # random kernel size
    @staticmethod
    def get_dynamic_padding():
        return random.choice((0, 1))  # random kernel size
# Inputs of the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
