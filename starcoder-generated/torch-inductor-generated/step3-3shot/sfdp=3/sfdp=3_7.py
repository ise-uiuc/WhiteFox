
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mat1 = torch.nn.Linear(512, 512)
        self.mat2 = torch.nn.Linear(512, 512)
        self.mat3 = torch.nn.Linear(512, 512)
        self.mat5 = torch.nn.Linear(512, 512)
        self.mat6 = torch.nn.Linear(512, 512)

    def forward(self, xin1):
        xout1 = self.mat1(xin1)
        xout2 = self.mat2(xout1)
        qk = xout1.matmul(xout2.transpose(-2, -1))
        qk2 = self.mat3(qk)
        qk3 = self.mat5(qk2)
        qk4 = self.mat6(qk3)
        vout = xout2.matmul(qk4)
        return vout

# Initializing the model
m = Model()

# Inputs to the model
xin1 = torch.randn(1, 512)
