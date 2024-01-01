
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(3, 8, 5, padding=2, padding_mode='zeros')
    def forward(self, x):
        y = self.conv(x)
        z = F.batch_norm(y, None, None)
        return z
# inputs to the model
x = torch.randn(1, 3, 20, 20)
