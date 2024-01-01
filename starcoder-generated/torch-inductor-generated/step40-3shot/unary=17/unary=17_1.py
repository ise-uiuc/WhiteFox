
class Model(torch.nn.Module):
    def __init__(self):
        self.a1 = torch.reshape(997, (1,1))
        self.a2 = torch.eye(1)
        self.a3 = torch.eye(3, 4)
        self.a4 = torch.rand(2,3)
        self.a5 = torch.rand(2,3,5)
        self.a6 = torch.rand(2,3,5,4)
        self.a7 = torch.rand(1,3,4,5,6,7)
        self.a8 = torch.rand(1,2,5,4)

        self.conv = torch.nn.ConvTranspose2d(3, 32, 1, stride=2)
        self.convm = torch.nn.ConvTranspose2d(3, 8, 1, stride=2, groups=3)
        self.convmb = torch.nn.ConvTranspose2d(32, 8, 3, stride=2, padding=1, groups=4)
        self.b1 = torch.randn(6,6)
        self.b2 = self.b1.view(3,2,4)
    def forward(self, x1):
        v1 = torch.sigmoid(self.conv(x1))
        v2 = torch.sigmoid(self.convm(x1))
        v3 = torch.sigmoid(self.convmb(v1))
        v4 = x1*x1
        v5 = self.a1*self.a2*self.a3*self.a4*self.a5*self.a6*self.a7*self.a8
        v6 = x1+x1
        v7 = v5+v6
        v8 = torch.sum(self.b2)
        v9 = torch.matmul(self.b1, self.b2)
        return v1*v2*v3*v4*v7*v8*v9
# Inputs to the model
x1 = torch.zeros(1, 3, 34, 38)
