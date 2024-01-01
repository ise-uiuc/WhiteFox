
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, (5, 5), stride=(2, 2), padding=(0, 0)) 
        self.conv2 = torch.nn.Conv2d(1, 1, (5, 5), stride=(2, 2), padding=(1, 1))
        self.dropout = torch.nn.Dropout(0.2)
        self.flatten = torch.nn.Flatten()
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.dropout(v1)
        v3 = self.conv2(v2)
        v4 = torch.sigmoid(v3)
        v5 = self.flatten(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 1, 256, 256)
