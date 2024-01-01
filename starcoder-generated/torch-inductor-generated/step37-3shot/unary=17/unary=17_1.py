
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convt1 = torch.nn.ConvTranspose2d(3, 1, kernel_size=(10, 1), stride=1)
        self.convt2 = torch.nn.ConvTranspose2d(1, 3, kernel_size=(5, 5), padding=(3, 2), stride=2)
        self.max_pool = torch.nn.MaxPool2d(4, 2, padding=0)
    def forward(self, x1):
        v1 = self.convt1(x1)
        v2 = self.convt2(v1)
        v3 = self.max_pool(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
