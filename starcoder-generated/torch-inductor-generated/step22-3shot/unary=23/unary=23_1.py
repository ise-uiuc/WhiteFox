
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.modules.batchnorm.BatchNorm2d(42, eps=1.4210854715202004e-05, momentum=0.1, affine=True, track_running_stats=True)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 42, 120, 90)
