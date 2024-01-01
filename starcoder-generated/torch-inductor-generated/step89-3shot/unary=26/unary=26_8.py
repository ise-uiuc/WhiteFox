
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_t = torch.nn.ConvTranspose2d(3848, 192, 1, stride=1, padding=0, bias=False)
        self.conv_t_bn = torch.nn.BatchNorm2d(192)
        self.conv_t1 = torch.nn.ConvTranspose2d(192, 192, 3, stride=3, padding=0, output_padding=0, groups=58, bias=False)
        self.conv_t1_bn = torch.nn.BatchNorm2d(192)
        self.conv_t2 = torch.nn.ConvTranspose3d(192, 320, (1, 3, 4), stride=(1, 3, 4), padding=(0, 1, 2), output_padding=(0, 1, 2), bias=True)
        self.conv_t3 = torch.nn.ConvTranspose2d(752, 86, 4, stride=1, padding=2, output_padding=1, bias=True)
    def forward(self, x72, x80):
        t1 = self.conv_t(x80)
        t2 = self.conv_t_bn(t1)
        t3 = torch.nn.functional.relu(t2)
        t4 = self.conv_t1(x72)
        t5 = self.conv_t1_bn(t4)
        t6 = torch.nn.functional.relu(t5)
        t7 = self.conv_t2(t3)
        t8 = t7 * -0.00485049801708202
        t9 = torch.where(t6 > 0, t7, t8)
        t10 = torch.where(t6 > 0, t9, t6)
        return torch.nn.functional.max_pool2d(t10, (1, 1))
# Inputs to the model
x72 = torch.randn(45, 752, 1, 28)
x80 = torch.randn(45, 3848, 5, 1)
