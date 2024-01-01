
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(***)
        self.relu0 = torch.nn.ReLU(***)
        self.conv1 = torch.nn.Conv2d(***)
        self.relu1 = torch.nn.ReLU(***)
        self.conv10 = torch.nn.Conv2d(***)
        self.relu1 = torch.nn.ReLU(***)
        self.conv11 = torch.nn.Conv2d(***)
        self.relu1 = torch.nn.ReLU(***)
        self.conv20 = torch.nn.Conv2d(***)
        self.relu2 = torch.nn.ReLU(***)
        self.conv21 = torch.nn.Conv2d(***)
        self.relu2 = torch.nn.ReLU(***)
        self.conv31 = torch.nn.Conv2d(***)
        self.relu3 = torch.nn.ReLU(***)
        self.conv41 = torch.nn.Conv2d(***)
        self.relu4 = torch.nn.ReLU(***)
        self.conv51 = torch.nn.Conv2d(***)
    def forward(self, x1):
        v1 = self.conv0(x1)
        v2 = self.relu0(v1)
        v3 = self.conv1(v2)
        v4 = self.relu1(v3)
        v5 = self.conv2(v4)
        v6 = self.relu2(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 2, 3, 4)
