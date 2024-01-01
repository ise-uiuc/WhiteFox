
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(3, 32, 3, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv1(x1)
        v3 = self.conv1(x1)
        v4 = self.conv2(x1)
        v1001 = self.conv2(x1)
        v1002 = self.conv2(x1)
        v1003 = self.conv2(x1)
        t1 = v1 + v2 + v3 + v4 + v1001 + v1002 + v1003
        v6 = self.conv1(x1)
        v7 = self.conv1(x1)
        v8 = self.conv1(x1)
        v9 = v6 + v7 + v8
        t2 = self.conv2(x1)
        v1004 = self.conv2(x1)
        v1005 = self.conv2(x1)
        v1006 = self.conv2(x1)
        v1007 = self.conv2(x1)
        t3 = v9 + v1004 + v1005 + v1006 + v1007
        v1008 = self.conv2(x1)
        v1009 = self.conv2(x1)
        v1010 = self.conv2(x1)
        v1011 = self.conv2(x1)
        v1012 = self.conv2(x1)
        v1013 = self.conv2(x1)
        t4 = v1008 + v1009 + v1010 + v1011 + v1012 + v1013
        v1014 = self.conv2(x1)
        v1015 = self.conv2(x1)
        v1016 = self.conv2(x1)
        v1017 = self.conv2(x1)
        v1018 = self.conv2(x1)
        v1019 = self.conv2(x1)
        t5 = v1014 + v1015 + v1016 + v1017 + v1018 + v1019
        t6 = torch.relu(t1 + t2 + t3 + t4 + t5)
        return t6
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
