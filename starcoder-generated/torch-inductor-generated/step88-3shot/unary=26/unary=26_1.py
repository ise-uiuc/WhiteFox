
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose3d(12, 4, (3,3,3), stride=2)
        self.conv = torch.nn.Conv3d(4, 2, (1,1,1), stride=1)
    def forward(self, x):
        u1 = self.conv_t(x)
        w1 = x.shape[2]//u1.shape[2]
        h1 = x.shape[3]//u1.shape[3]
        z1 = x.shape[4] //u1.shape[4]
        u2 = torch.max_unpool3d(u1, 2, (w1,h1,z1))
        u3 = u2 > torch.max(u1.shape[2], u1.shape[3], u1.shape[4])
        u4 = u2 * 0.52291
        w2 = u4 * u3
        u5 = torch.where(u3, u2, w2)
        u6 = torch.max_unpool3d(u5, 2, (w1,h1,z1))
        return u6
# Inputs to the model
x = torch.randn(1, 12, 20, 50, 30)
