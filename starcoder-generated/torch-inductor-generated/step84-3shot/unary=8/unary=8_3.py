
class Model(torch.nn.Module):
    def __init__(self):
        return torch.nn.Sequential(
            torch.nn.ConvTranspose2d(8, 16, 3, stride=2, padding=1, output_padding=1),
            torch.nn.AdaptiveAvgPool2d(3))
    def forward(self, input):
        x1 = self.layers(input)
        x1, _ = x1.split([2, 7], dim=2)
        x1, _ = x1.split([2, 4], dim=5)
        x2 = self.layers(x1)
        x2, _ = x2.split([1, 2], dim=-1)
        x2, _ = x2.split([2, 6], dim=-2)
        x3, _ = x2.split([2, 2], dim=3)
        x4, _ = x3.split([1, 1], dim=-1)
        x5 = torch.nn.functional.adaptive_avg_pool2d(x4, 1)
        x6 = x5.view(-1)
        return x6
# Inputs to the model
input = Variable(torch.randn(7, 8, 20, 35))
