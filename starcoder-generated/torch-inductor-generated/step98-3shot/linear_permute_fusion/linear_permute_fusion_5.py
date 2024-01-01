
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.v1 = torch.nn.Conv3d(2, 4, 2)
        self.v2 = torch.nn.functional.interpolate(self.v1, scale_factor=self.v1.stride)
    def forward(self, x2):
        v0 = x2
        v9 = self.v1(v0)
        self.v2 = torch.nn.functional.interpolate(v9, scale_factor=self.v1._output_size)
        return self.v1._output_size
# Inputs to the model
x2 = torch.randn(5, 2, 3, 2, 3)
