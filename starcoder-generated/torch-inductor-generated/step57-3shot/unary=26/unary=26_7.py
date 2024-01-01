
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_t1 = nn.ConvTranspose2d(10, 20, 3, stride=1, padding=2, output_padding=2)
        self.conv_t2 = nn.ConvTranspose2d(20, 3, 3, stride=1)
    def forward(self, x22):
        x22 = self.conv_t1(x22)
        x22 = x22 > 0
        x22 = x22 * 1
        x22 = torch.where(x22, x22, x22)
        x22 = self.conv_t2(x22)
        return x22
# Input to the model
x22 = torch.randn(1, 10, 16, 19)
