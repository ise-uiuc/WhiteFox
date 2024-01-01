
class Model(torch.nn.Module):
    def __init__(self, min_value=[-0.3114414225, -8.686340339, -4.7155270571, -8.7270317078, 1.0554384441, -0.8102259665, -1.7250378657, 6.6969900131], max_value=[2.0911283789, -6.1379118442, 5.7366183758, 7.2606854916, 0.0433016797, 8.4918063164, -1.1726343899, 7.5745980263]):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose3d(4, 4, 1, stride=1, padding=1)
        self.dropout = torch.nn.Dropout3d(p=0.21)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        # Flatten input to 2D.
        x2 = x1.permute(0, 2, 3, 4, 1).contiguous().view(x1.size(0), x1.size(2), x1.size(3), x1.size(4) * x1.size(1))
        x3 = self.dropout(self.conv_transpose(x2))
        # Unflatten back to 5D.
        x4 = x3.view(x3.size(0), x1.size(2), x1.size(3), x1.size(4), x1.size(1))
        x5 = x4.permute(0, 4, 1, 2, 3).contiguous()
        x6 = torch.clamp_min(x5, self.min_value)
        x7 = torch.clamp_max(x6, self.max_value)
        return x7
# Inputs to the model
x1 = torch.randn(4, 8, 32, 32, 32)
