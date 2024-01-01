
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv_t = nn.ConvTranspose2d(57, 9, (23, 52), stride=(15, 8), padding=(12, 11), dilation=(2, 1))
    def forward(self, x5):
        p1 = self.conv_t(x5)
        p2 = p1 > 0.0
        p3 = p1 * 0.25
        p4 = torch.where(p2, p1, p3)
        p5 = torch.sigmoid(p4)
        return torch.cat((p5[:, :, 0:2, 1:1], p4[:, :, 2:2, :]), 1)
# Inputs to the model
x5 = torch.randn(6, 57, 89, 52)
