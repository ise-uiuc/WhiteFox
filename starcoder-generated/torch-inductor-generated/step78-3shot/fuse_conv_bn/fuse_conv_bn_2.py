
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.batch_norm = nn.BatchNorm2d(100)
        self.conv = nn.Conv2d(100, 200, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x):
        x = F.relu(x)
        x = self.conv(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        return x
# Inputs to the model
x = torch.randn(1, 100, 28, 28)
