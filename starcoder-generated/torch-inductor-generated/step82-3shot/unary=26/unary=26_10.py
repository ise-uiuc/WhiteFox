
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(4, 1, 4, stride=4, padding=1)
    def forward(self, x19):
        x20 = self.conv_t(x19)
        x21 = x20 > 0
        x22 = x20 * -0.13359355676651
        x23 = torch.where(x21, x20, x22)
        a24 = torch.add(1.4238811526204257, x22, alpha=1)
        a25 = torch.add(1.5058314327681624, a24, alpha=1)
        a26 = torch.add(0.8217762323322541, a25, alpha=1)
        x27 = torch.nn.functional.max_pool2d(a26, stride=7, kernel_size=(3, 9), padding=(1, 0))
        x28 = torch.nn.functional.adaptive_avg_pool2d(x27, (1, 1))
        x29 = torch.nn.functional.relu(x28)
        return torch.ceil(x29)
# Inputs to the model
x19 = torch.randn(15, 4, 3, 9)
