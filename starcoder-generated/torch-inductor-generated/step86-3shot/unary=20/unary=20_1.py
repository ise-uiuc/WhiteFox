
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(8, 64, kernel_size=(10, 7), stride=3, padding=(0, 0))
        self.pad = torch.nn.ReplicationPad2d(1)
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = self.pad(v1)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 8, 43, 13)
