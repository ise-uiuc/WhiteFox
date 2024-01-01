
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.modules.ModuleList([
            torch.nn.Linear(20, 30),
            torch.nn.Tanh(),
            torch.nn.Linear(30, 40),
        ])
        self.conv=torch.nn.Conv2d(20, 40, kernel_size=1, padding=0)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        for i in range(0, 3):
            l = self.layer[i]
            x = l(x)
            #x = self.sigmoid(x)
        conv = self.conv(x)
        tanh = torch.tanh(conv)
        return tanh
# Inputs to the model
x = torch.randn(1, 20, 1, 1)
