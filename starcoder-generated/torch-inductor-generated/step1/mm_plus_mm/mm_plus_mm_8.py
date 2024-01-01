
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 3, stride=1, padding=1)
        self.fc = torch.nn.Linear(8 * 4 * 4, 64)
 
    def forward(self, input, filter):
        x = self.conv(input)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = torch.mm(x, filter)
        return x