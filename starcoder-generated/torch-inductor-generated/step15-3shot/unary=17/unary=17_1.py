
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.ConvTranspose2d(3, 8, kernel_size=(2, 2), stride=(4, 4), padding=(1, 1), output_padding=(2, 2))
    def forward(self, x1):
        v1 = self.layer1(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 4, 4)
