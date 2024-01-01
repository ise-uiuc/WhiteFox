
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1, bias=False)
    def forward(self, x3):
        x = x3
        x4 = self.conv_transpose(x)
        x5 = x4 * 0.45340277777510936
        x6 = x4 * x4 * x4
        x7 = x6 * 0.3092516939403577
        x8 = x5 + x7
        x10 = x8 * 0.9742443818094455
        x11 = torch.tanh(x10)
        v2 = x4 * 0.5
        v3 = x4 * x4 * x4
        v4 = v3 * 0.044715
        v5 = x4 + v4
        v6 = v5 * 0.7978845608028654
        v7 = torch.tanh(v6)
        v8 = v7 + 1
        v9 = v2 * v8
        x12 = v9 * x11
        x13 = x12 + x
        x14 = x13 * 0.041447086679542867
        x15 = x13 * x13 * x13
        v10 = v3 * 0.009969108573734632
        v11 = x4 + v10
        v12 = v11 * 0.47464823447866365
        v13 = torch.tanh(v12)
        v14 = v2 * v13
        x16 = v14 * x11
        x17 = x16 + x14
        x18 = x17 * 0.021424672719869478
        x19 = x17 * x17 * x17
        v15 = x6 * 0.005670097458058505
        v16 = v3 + v15
        v17 = v16 * 0.45546406999857767
        v18 = torch.tanh(v17)
        v19 = v18 * v18
        x20 = v19 * x11
        x21 = x20 + x18
        x22 = x21 * 0.011580318898303273
        x23 = x21 * x21 * x21
        v20 = x7 * -3.5227615019248454e-05
        v21 = v4 + v20
        v22 = v21 * 0.45543786498704396
        v23 = torch.tanh(v22)
        x24 = v19 * v23
        x25 = x24 + x22
        x26 = x25 * 0.01703979998934611
        x27 = x25 * x25 * x25
        v6 = x6 * 0.010074230784605682
        v7 = v3 + v6
        v8 = v7 * 0.20999840236517297
        v9 = torch.tanh(v8)
        x28 = v9 * v19
        x29 = x28 + x8
        x30 = x29 * 0.02289223018730019
        x31 = x29 * x29 * x29
        return x31
# Inputs to the model
x3 = torch.randn(1, 64, 56, 56)
