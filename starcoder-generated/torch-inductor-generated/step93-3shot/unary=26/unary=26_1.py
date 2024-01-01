
class MyModel(torch.nn.Module):
    def __init__(self, chn_in, num_filters):
        super().__init__()
        self.layer1 = torch.nn.Sequential(torch.nn.ConvTranspose2d(ch_in, num_filters, 2, stride=(2, 1), bias=False), torch.nn.ConvTranspose2d(num_filters, num_filters, 1, stride=1, padding=0, bias=False), torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(num_filters, num_filters, 9, stride=2, padding=4, dilation=2, bias=False), torch.nn.BatchNorm2d(num_filters), torch.nn.ReLU())
        self.layer3 = torch.nn.Sequential(torch.nn.Conv2d(num_filters, num_filters, 1, stride=1, padding=0, bias=False), torch.nn.ReLU())
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
# Inputs to the model
chn_in, num_filters = 10, 20
x = torch.rand([1, chn_in, 100, 40])
