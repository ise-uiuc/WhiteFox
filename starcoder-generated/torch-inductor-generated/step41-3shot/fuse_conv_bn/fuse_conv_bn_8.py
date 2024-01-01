
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        torch.manual_seed(0)
        self.conv1 = torch.nn.Conv3d(4, 3, (3,2,2), stride=1, padding=(1,2,1))
        torch.manual_seed(1)
        self.conv2 = torch.nn.Conv3d(3, 5, (3,2,1), stride=1, padding=(1,1,0))
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

