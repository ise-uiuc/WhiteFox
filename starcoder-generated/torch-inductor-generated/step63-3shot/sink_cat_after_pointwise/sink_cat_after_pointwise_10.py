
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer2 = torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer3 = torch.nn.Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
# Inputs to the model
x = torch.randn(1, 50, 100, 200)
