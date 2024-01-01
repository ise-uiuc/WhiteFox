
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        d = 0.3
        self.sconv1 = torch.nn.Conv2d(3, 3, 3, stride=2, padding=1)
        self.sconv2 = torch.nn.Conv2d(3, 3, 3, stride=2, padding=1)
        self.sconv3 = torch.nn.Conv2d(3, 3, 3, stride=2, padding=1)
    def forward(self, x1, x2, x3, x4):
        v1 = self.sconv1(x3)
        v2 = torch.nn.LeakyReLU(d)(v1)
        v3 = torch.matmul(v2, x2)
        v4 = torch.transpose(v3, 1, 2)
        v5 = self.sconv3(v4)
        v6 = torch.nn.LeakyReLU(d)(v5)
        v7 = self.sconv2(v6)
        v8 = torch.nn.Softmax2d()(v7)
        v9 = torch.matmul(v8, x1)
        v10 = self.sconv2(v9)
        v11 = torch.nn.ReLU()(v10)
        tmp0 = Variable(torch.FloatTensor([[[3.71901048]]]))
        v12 = torch.add(v11, tmp0)
        v13 = torch.sub(v11, tmp0)
        v14 = torch.mul(v11, tmp0)
        tmp1 = Variable(torch.FloatTensor([[[0]]]))
        v15 = torch.add(v11, tmp1)
        tmp2 = Variable(torch.FloatTensor([[[4]]]))
        v16 = torch.add(v11, tmp2)
        tmp3 = Variable(torch.FloatTensor([[[3]]]))
        v17 = torch.add(v11, tmp3)
        v18 = torch.nn.Sigmoid()(v12)
        v19 = torch.nn.Tanh()(v12)
        v20 = torch.tanh(v13)
        v21 = torch.sigmoid(v15)
        v22 = torch.sigmoid(v16)
        v23 = torch.tanh(v17)
        b0 = torch.matmul(x4, v18)
        v24 = torch.transpose(b0, 1, 2)
        b1 = torch.matmul(x4, v19)
        v25 = torch.transpose(b1, 1, 2)
        b2 = torch.matmul(x4, v20)
        v26 = torch.transpose(b2, 1, 2)
        b3 = torch.matmul(x4, v21)
        v27 = torch.transpose(b3, 1, 2)
        b4 = torch.matmul(x4, v22)
        v28 = torch.transpose(b4, 1, 2)
        b5 = torch.matmul(x4, v23)
        v29 = torch.transpose(b5, 1, 2)
        v30 = np.array([[[1, 1, 0.990417257353301]])]
        tmp5 = Variable(torch.FloatTensor(v30))
        v31 = torch.add(v3, tmp5)
        v32 = torch.div(v3, tmp5)
        v33 = torch.div(v3, tmp0)
# Inputs to the model
x1 = torch.randn(1, 9)
x2 = torch.randn(1, 3, 3)
x3 = torch.randn(1, 3, 64, 64)
x4 = torch.randn(1, 9)
