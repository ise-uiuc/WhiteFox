
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample = torch.nn.Upsample(mode='linear', scale_factor=None, align_corners=None)
    def forward(self, x, target):
        var = self.upsample(x, output_size=target.shape)
        output = var + target
        return output

# Inputs to the model
x = torch.randn(1, 1, 2, 2, requires_grad=True)
target = torch.randn(2, 4, 2, 2)
