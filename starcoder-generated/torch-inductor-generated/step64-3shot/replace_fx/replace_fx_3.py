
class model_new(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 5, 5)
        self.gru = torch.nn.GRU(5, 5)
        self.batchnorm = torch.nn.BatchNorm2d(5)
    def forward(self, x):
        x = F.dropout(x)
        x = x.transpose(2, 3)
        x1 = self.conv(x)
        x2 = torch.rand_like(x1)
        x3 = self.batchnorm(x2)
        x4 = self.gru(x3)
        x5 = torch.rand_like(x4)
        x6 = self.batchnorm(x5)
        x7 = torch.nn.functional.dropout(x6, p=0.5)
        return x7
# Inputs to the model
x = torch.randn(1, 4, 8, 8)
