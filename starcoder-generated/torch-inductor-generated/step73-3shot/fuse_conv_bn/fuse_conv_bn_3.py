
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(4, 4, (3, 3), padding=(1, 1))
        self.relu = torch.nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv1(x)
        x = torch.transpose(x, 2, 3)
        x = self.relu(x)
        return x.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
# Inputs to the model
x = torch.randn(1, 4, 10, 10)
