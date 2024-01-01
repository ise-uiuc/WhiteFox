
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 16, 1, stride=1, padding=1)
        self.linear1 = torch.nn.Linear(192, 1024)
        self.linear2 = torch.nn.Linear(1024, 256)
 
    def forward(self, x1):
        r = self.conv1(x1)
        x2 = self.conv2(x1)
        r = F.max_pool2d(r, 2)
        x2 = F.max_pool2d(x2, 2)
        c2 = torch.cat([r, x2], 1)
        c2 = torch.flatten(c2, 1)
        