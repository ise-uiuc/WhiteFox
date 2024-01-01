
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.con_t1 = torch.nn.ConvTranspose2d(in_channels=1, out_channels=2, bias=False, kernel_size=(4, 4), stride=1, padding=2, output_padding=0)
        self.sigmoid = torch.nn.Sigmoid()
        self.con_t2 = torch.nn.ConvTranspose2d(in_channels=1, out_channels=2, bias=False, kernel_size=(6, 1), stride=1, padding=0, output_padding=0)
    def forward(self, x1):
        v1 = self.con_t1(x1)
        v2 = self.sigmoid(v1)
        v3 = self.con_t2(v2)
        v4 = self.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 16, 16)
