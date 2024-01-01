
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.module = torch.nn.quantized.modules.conv.Conv2d(2, 3, 3, stride=1, padding=1)
    def get_weight_count(self):
        return torch.quantization.fuse_modules(self, ['module.weight'], inplace=False)
    def forward(self, x1):
        t1 = self.module(x1)
        t2 = torch.cumsum(t1, 1)
        return t2
# Inputs to the model
x1 = torch.randn(4, 2, 128, 128, device='cuda:0')
