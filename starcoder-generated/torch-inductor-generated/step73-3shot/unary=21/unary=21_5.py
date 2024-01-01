
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conva = torch.nn.Conv2d(3, 16, kernel_size=5, padding=0)
        self.convb = torch.nn.Conv2d(16, 8, kernel_size=3, padding=0)
        self.convc = torch.nn.Conv2d(8, 4, kernel_size=3, padding=0)
        self.convd = torch.nn.Conv2d(4, 8, kernel_size=3, padding=0)
        self.conve = torch.nn.Conv2d(8, 2, kernel_size=3, padding=0)
        self.convf = torch.nn.Conv2d(2, 1, kernel_size=3, padding=0)
    def forward(self, x):
        v1 = self.conva(x)
        v2 = torch.tanh(v1)
        v3 = self.convb(v2)
        v4 = torch.tanh(v3)
        v5 = self.convc(v4)
        v6 = torch.tanh(v5)
        v7 = self.convd(v6)
        v8 = torch.tanh(v7)
        v9 = self.conve(v8)
        v10 = torch.tanh(v9)
        v11 = self.convf(v10)
        v12 = torch.tanh(v11)
        return v12.detach()
# Inputs to the model
x = torch.randn(1, 3, 224, 224)
