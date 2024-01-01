
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.__a = torch.nn.ReLU(inplace=True)
        self.a = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.__a(x1)
        v2 = self.a(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
