
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)
        self.bn1 = torch.nn.BatchNorm2d(3)
        self.bn2 = torch.nn.BatchNorm2d(3)
        self.conv2 = torch.nn.Conv2d(3, 1, 1)
        self.softmax = torch.nn.Softmax(dim=1)
        self.drop = torch.nn.Dropout(0.5)

    def forward(self, x1):
        # Conv1 and bn1 are optimized.
        s = self.conv(x1)
        t1 = self.bn1(s)
        # Conv 2 and bn2 are not optimized as they have different
        # behavior at training and inference time.
        t2 = self.conv(s)
        t2 = self.bn2(t2)
        y1 = self.conv2(t1)
        y2 = self.conv2(t2)
        self.drop(y1)
        self.drop(y2)
        self.softmax(y1)
        self.softmax(y2)
        return y1, y2
# Inputs to the model
x1 = torch.randn(1, 3, 4, 4)
