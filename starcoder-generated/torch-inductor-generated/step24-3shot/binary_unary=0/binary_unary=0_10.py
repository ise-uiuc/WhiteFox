
class Model(torch.nn.Module):
    def forward(self, x1):
        v1 = x1.view(1, 256, 1, 1)
        v2 = torch.nn.functional.max_pool2d(v1, 1, stride=1, padding=0, dilation=1, ceil_mode=False)
        v3 = v2.permute(0, 2, 3, 1)
        return v3
# Inputs to the model
x1 = torch.randn((1, 64))
