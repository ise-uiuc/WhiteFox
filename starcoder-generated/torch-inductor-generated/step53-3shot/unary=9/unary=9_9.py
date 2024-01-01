
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm([2, 2], elementwise_affine=False)
    def forward(self, x1):
        v1 = self.layer_norm(x1)
        v2 = torch.nn.functional.conv2d(v1, torch.ones(1, 1, 3, 3), padding=1)
        v3 = v2.clamp(min=0, max=6)
        v4 = v3 / 6
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 2, 2)
