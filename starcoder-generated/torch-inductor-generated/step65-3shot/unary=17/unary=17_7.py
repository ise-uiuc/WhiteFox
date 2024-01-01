
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convt1 = torch.nn.ConvTranspose1d(64, 64, 1, 1, 0, 1, bias=False)
        self.convt2 = torch.nn.ConvTranspose1d(64, 64, 1, 1, 0, 1, bias=False)
        self.convt3 = torch.nn.ConvTranspose1d(64, 64, 1, 1, 0, 1)
        self.max_pool1 = torch.nn.MaxPool1d(3, 3, 1)
        self.convt4 = torch.nn.ConvTranspose1d(64, 1, 3, 1, 0, 1)
        self.max_pool2 = torch.nn.MaxPool1d(32, 32, 1)

    def forward(self, x1):
        v1 = self.convt1(x1)       
        v10 = torch.relu(v1)
        v2 = self.convt2(v10)       
        v20 = torch.relu(v2)
        v3 = self.convt3(v20)       
        v30 = torch.relu(v3)
        v4 = self.max_pool1(v30)
        v5 = self.convt4(v4)
        v50 = torch.relu(v5)
        v6 = self.max_pool2(v50)
        v7 = torch.sigmoid(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 64, 32)
