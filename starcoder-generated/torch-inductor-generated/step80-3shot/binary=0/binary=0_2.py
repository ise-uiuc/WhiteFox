
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convt_ = torch.nn.ConvTranspose2d(2, 2, 3, stride=1, padding=1, bias=False)
    def forward(self, x1):
        v1 = self.convt_(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 2, 12, 12)
