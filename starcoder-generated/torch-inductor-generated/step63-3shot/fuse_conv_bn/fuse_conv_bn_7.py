
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 1, bias=False)
        self.conv2 = torch.nn.Conv2d(3, 3, 1, bias=False)
        self.conv3 = torch.nn.Conv2d(3, 3, 1, 1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(1, affine=False)
        self.bn1b = torch.nn.BatchNorm2d(1, affine=False)
        self.bn2 = torch.nn.BatchNorm2d(1, affine=False)
        self.bn1c = torch.nn.BatchNorm2d(1, affine=False)
    def forward(self, input):
        conv1 = self.conv1(input) # conv1 = input
        conv1b = self.conv1(conv1) # conv1b = input
        conv1c = self.conv1(input) # conv1c = input
        conv1d = self.conv1(torch.transpose(torch.add(conv1b, conv1c), 1, 3)) # conv1d = input
        bn1 = self.bn1(conv1d)
        bn1b = self.bn1(self.bn1b(conv1d))
        bn1c = self.bn1(self.bn1b(conv1d))
        bn1d = self.bn1(torch.add(conv1d, self.bn1b(conv1d)))
        bn2 = self.bn1(torch.transpose(torch.add(bn1, bn1b), 1, 3))
        bn2b = self.bn1b(torch.transpose(torch.add(bn1b, bn1c), 1, 3))
        bn2c = self.bn1b(torch.transpose(torch.add(bn1c, bn1d), 1, 3))
        bn2d = self.bn1b(torch.transpose(torch.add(bn1d, bn1d), 1, 3))
        return bn2
# Inputs to the model
input = torch.randn(2, 3, 8, 8)
