
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.nn.functional.softmax(x1, dim=-1)
        x3 = F.pixel_shuffle(x2, upscale_factor=2)
        return (x2, x3)
# Inputs to the model
x1 = torch.randn(1, 12, 128, 16)
