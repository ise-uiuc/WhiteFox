
class model(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, 1)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        return self.dropout(self.conv1(x))
