
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.conv = torch.nn.Conv2d(1, 1, (1, 1))
    def forward(self, x1):
        x1 = self.sigmoid(x1)
        x1 = self.conv(x1).permute(2, 0, 3, 1)
        x1 = x1.reshape(1, 2)
        return x1
# Inputs to the model
x1 = torch.randn([1, 2, 2, 1])
