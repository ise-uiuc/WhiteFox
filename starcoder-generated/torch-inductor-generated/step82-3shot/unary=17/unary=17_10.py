
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 3, [4, 3], padding=[4, 3])
        self.conv1 = torch.nn.ConvTranspose2d([199, 4], 3, [3, [1, 2, 3]], stride=[5, 6], padding=[7, [[[7, 7], [6, 6], [5, 5]], [[6, 6], [5, 5], [4, 4]], [[5, 5], [4, 4], [3, 3]]]], output_padding=[8, [[[17, 17], [16, 16], [15, 15]], [[16, 16], [15, 15], [14, 14]], [[15, 15], [14, 14], [13, 13]]]])
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = self.conv1(v2)
        v4 = torch.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 4, 199, 4)
