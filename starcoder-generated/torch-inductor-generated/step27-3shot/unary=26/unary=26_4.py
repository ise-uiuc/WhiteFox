
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_42 = torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(1, 2), stride=(1, 2), groups=2,)
        self.conv2d_5689 = torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(1, 2), stride=(1, 2), groups=2,)
    def forward(self, x):
        y = self.conv2d_42(x)
        p = torch.flatten(y, start_dim=2)
        r = torch.flatten(y, start_dim=2)
        s = p - r
        t = s + y
        u = self.conv2d_5689(t)
        return torch.flatten(t, start_dim=2)
# Inputs to the model
x = torch.randn(1, 2, 6, 6)
