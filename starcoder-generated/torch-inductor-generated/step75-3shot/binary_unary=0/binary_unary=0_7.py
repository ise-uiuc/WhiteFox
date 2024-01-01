
class Model(torch.nn.Module):
    def __init__(self):
        self.conv11 = nn.Conv2d(1, 8, 3, padding='same')
        self.conv12 = nn.Conv2d(1, 16, 5, padding='same')
        self.conv13 = nn.Conv2d(1, 32, 7, padding='same')
        self.conv2 = Conv2d(16, 32, 3, True, True, 1, 1)
    def forward(self, x):
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv2(x)
        return x
# Input to the model
x = torch.rand(1, 1, 1, 1)
