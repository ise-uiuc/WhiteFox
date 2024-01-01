
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        num_classes = 2622
        self.num_classes = num_classes
        self.conv0 = nn.Conv2d(3, self.num_classes*16, 3,
                               stride=1, padding=1)
        self.blocks = nn.Sequential()
        self.conv1 = nn.Conv2d(3, 64, 3,
                               stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, self.num_classes*16, 1,
                               stride=1, padding=0)
        self.blocks.add_module('block1', self._make_layer())
        self.conv3 = nn.Conv2d(3, 64, 3,
                               stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(128, self.num_classes*16, 1,
                               stride=1, padding=0)
        self.blocks.add_module('block1', self._make_layer())
        self.conv5 = nn.Conv2d(3, 64, 3,
                               stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(128, self.num_classes*16, 1,
                               stride=1, padding=0)
        self.blocks.add_module('block1', self._make_layer())
    def _make_layer(self):
        layers = []
        layers.apdend(self._make_block())
        return nn.Sequential(*layers)
    def _make_block(self):
        layer1 = self.conv1(3, 64, 1, 1, 0)
        layer2 = self.bn1(layer1)
        layer3 = self.relu(layer2)
        layer4 = self.conv2(layer3)
        return layer4
    def _make_layer2(self):
        layers = []
        layers.apdend(self._make_block())
        return nn.Sequential(*layers)
    def forward(self, x):
        v1 = self.conv0(x)
        v2 = v1.reshape(x.shape[0], self.num_classes, 16, x.shape[3], x.shape[4])
        v3 = v2.permute([0, 1, 3, 4, 2])
        h1 = self.relu(self.bn1(self.conv1(x)))
        h2 = self.relu(self.bn2(self.conv3(h1)) + \
            self.conv4(self.relu(self.bn2(self.conv3(h1)))))
        h3 = self.relu(self.bn3(self.conv5(x)) + \
            self.conv6(self.relu(self.bn3(self.conv5(x)))))
        v4 = torch.cat([v3, h2, h3], 4)
        x = self.blocks(v4)
        return self.relu(self.conv7(x))
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
