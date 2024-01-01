
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=13, out_channels=18, kernel_size=(2, 3), stride=3, padding=(0, 1))
        self.conv2 = torch.nn.Conv2d(in_channels=18, out_channels=9, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv3 = torch.nn.Conv2d(in_channels=9, out_channels=13, kernel_size=(1, 5), stride=1, padding=(0, 2))
        self.avgpool = torch.nn.AvgPool2d(kernel_size=(4, 3), stride=(1, 3), padding=(2, 2))
        self.adaptive_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = torch.nn.Linear(in_features=192, out_features=32, bias=True)
    def forward(self, input0):
        