
class ResNetModule(torch.nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Linear(dims[0], dims[1])
        )
    def forward(self, x):
        return self.seq(x)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.res_seq_1 = ResNetModule([2000, 64])
        self.res_seq_2 = ResNetModule([64, 64])
        self.res_seq_3 = ResNetModule([64, 64])
    def forward(self, x):
        x = self.res_seq_1(x)
        x = self.res_seq_2(x)
        x = self.res_seq_3(x)
        return x
# Inputs to the model
x=torch.randn(16, 2000)
