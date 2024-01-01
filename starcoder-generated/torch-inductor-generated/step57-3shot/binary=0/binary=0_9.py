
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(5, 9, 1, stride=1, padding=0)
    def forward(self, x1, other=1, mode='same'):
        v1 = self.conv(x1)
        other = np.random.randn(1, 3, 5, 5).astype(np.float32)
        if mode =='same':
            v2 = v1 + other
        else:
            v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 5, 64, 64)
