
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_t = torch.nn.Sequential(torch.nn.ConvTranspose2d(139, 417, 11, stride=2, padding=5, output_padding=1, bias=True),torch.nn.ReLU(),torch.nn.ConvTranspose2d(417, 305, 11, stride=1, padding=4, bias=True),torch.nn.ReLU(),torch.nn.ConvTranspose2d(305, 146, 10, stride=1, padding=2, bias=True))
    def forward(self, x27):
        t1 = self.conv_t(x27)
        t2 = t1 > 0
        t3 = t1 * 0.248297
        t4 = torch.where(t2, t1, t3)
        return t4 + torch.nn.functional.adaptive_avg_pool2d(t4, (1, 1))
# Inputs to the model
x27 = torch.randn(18, 139, 50, 25)
