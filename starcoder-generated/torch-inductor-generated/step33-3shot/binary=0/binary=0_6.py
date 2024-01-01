
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conva = torch.nn.Conv2d(3, 8, (1, 3), (1, 1), (0, 1))
        self.convb = torch.nn.Conv2d(8, 4, (7, 1), (1, 1), (0, 1))
    def forward(self, x1, other=None):
        var1 = self.conva(x1)
        var2 = self.convb(var1)
        if other == None:
            other = torch.randn([var2.shape[0]])
        var3 = var2 + other
        return var3
# Inputs to the model
x1 = torch.randn(3, 3, 255, 255)

