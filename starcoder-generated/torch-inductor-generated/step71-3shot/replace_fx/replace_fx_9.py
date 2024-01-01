
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=2, out_channels=4, kernel_size=2, stride=2, padding=2, dilation=2)
        self.conv2 = torch.nn.Conv2d(in_channels=4, out_channels=8, kernel_size=2, stride=2, padding=2, dilation=2)
        self.conv3 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=2, stride=2, padding=2, dilation=2)
        self.fc1 = torch.nn.Linear(in_features=4*4*16, out_features=64)
        self.dropout = torch.nn.Dropout(0.05)
    def forward(self, x):
        x = self.fc1(self.dropout(self.conv3(self.dropout(self.conv2(self.dropout(self.conv(x), p=0.4), p=0.5)))))
        return x
# Inputs to the model
x = torch.randn(1, 2, 2, 2)
