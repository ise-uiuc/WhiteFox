
class SubModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(5, 10, 3, stride=2, padding=1)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(10, 10, 3, stride=2, padding=1, output_padding=1)
    def forward(self, x1):
        v = self.conv_transpose1(x1)
        v = self.conv_transpose2(v)
        return v
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sub_model = SubModel()
    def forward(self, x1):
        v = self.sub_model.conv_transpose(x1)
        return v
# Inputs to the model
x1 = torch.randn(1, 5, 64, 64)
