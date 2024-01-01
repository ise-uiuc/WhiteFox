
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1024, 640, 1, stride=1, padding=0)
        self.conv_1 = torch.nn.Conv2d(384, 640, 9, stride=4, padding=0)
        self.conv_2 = torch.nn.Conv2d(21, 70, 7, stride=3, padding=3)
        self.conv_3 = torch.nn.Conv2d(18, 2, 3, stride=2, padding=0)
    def forward(self, input, input_6, input_5, input_8, input_4):
        v2 = self.conv(input_4)
        v3 = self.conv_1(input_8)
        v4 = v3 * v3
        v14 = self.conv_2(input_6) 
        v5 = v14 * 0.5
        v6 = v14 * v14
        v45 = self.conv_3(input_5)
        v7 = v6 * v14
        v8 = v6 * 0.044715
        v9 = v7 + v8
        v10 = v3 * v6
        v11 = v10 * 0.7978845608028654
        v12 = torch.tanh(v11)
        v13 = v12 + 1
        v15 = v5 * v13
        v16 = v4 * v13
        v17 = v45 * v16
        v19 = v17 * v15
        t1 = v2 + v2
        t2 = t1 * 0.5
        t3 = t1 * t1
        t4 = t3 * t1
        t5 = t4 * 0.044715
        t6 = t1 + t5
        t7 = t6 * 0.7978845608028654
        t8 = torch.tanh(t7)
        t9 = t8 + 1
        t10 = t2 * t9
        t11 = t10 * 0.7978845608028654
        t12 = torch.tanh(t11)
        t13 = t12 + 1
        t14 = t13 * t2
        return t14
# Inputs to the model
input = torch.randn(1, 1024, 19, 19)
input_6 = torch.randn(1, 384, 49, 49)
input_5 = torch.randn(1, 21, 32, 32)
input_8 = torch.randn(1, 18, 62, 50)
input_4 = torch.randn(1, 2, 9, 3)
