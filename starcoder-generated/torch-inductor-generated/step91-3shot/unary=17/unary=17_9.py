
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.convtranpose = torch.nn.ConvTranspose2d(1, 1, kernel_size=3, stride=3, padding=1, output_padding=1)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.convtranpose(x1)
        v2 = self.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 16, 16)
