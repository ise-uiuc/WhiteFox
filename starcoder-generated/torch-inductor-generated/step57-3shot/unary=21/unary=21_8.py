
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_norm = torch.nn.BatchNorm2d(64, affine=False)
    def forward(self, x):
        v1 = self.batch_norm(x)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x = torch.randn(64, 64, 256, 256)
