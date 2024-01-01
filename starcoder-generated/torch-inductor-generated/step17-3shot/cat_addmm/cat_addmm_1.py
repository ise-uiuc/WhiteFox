
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers1 = nn.Linear(4, 4)
        self.layers2 = nn.Linear(4, 18)
        self.layers3 = nn.Linear(18, 2)
        self.layers4 = nn.Linear(2, 6)
    def forward(self, x):
        x = self.layers1(x)
        x = F.relu(x, inplace=False)
        x = torch.cat((x, x), dim=1)
        x = F.relu(x, inplace=False)
        x = torch.cat((x, x, x, x, x, x), dim=1)
        x = F.relu(x, inplace=False)
        x = self.layers2(x)
        x = F.relu(x, inplace=False)
        x = self.layers3(x)
        x = F.max_pool2d(x, kernel_size=2, stride=1)
        x = self.layers4(x)
        x = F.max_pool2d(x, kernel_size=2, stride=1)
        return x
# Inputs to the model
x = torch.randn(2, 4)
