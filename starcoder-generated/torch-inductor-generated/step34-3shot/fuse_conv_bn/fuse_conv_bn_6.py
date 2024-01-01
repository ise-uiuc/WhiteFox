
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(4, 4, 5)
        self.batchnorm1d = torch.nn.BatchNorm1d(4)
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
        self.con2 = torch.nn.Conv2d(4, 3, 5)
    def forward(self, x1):
        s = self.conv1(x1)
        t = s.transpose(1,2)
        r1 = self.relu1(s)
        bn1 = self.batchnorm1d(s)
        y = r1 + bn1
        r2 = self.relu2(y)
        z = self.con2(r2.unsqueeze(0).unsqueeze(0))
        return z.squeeze(0).squeeze(0)
# Inputs to the model
x1 = torch.randn(1, 4, 20)
