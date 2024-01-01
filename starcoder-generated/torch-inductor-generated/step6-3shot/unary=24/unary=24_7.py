
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv = torch.nn.Conv2d(22, 40, 5, stride=1, groups=3, padding=1, bias=True)
        self.negative_slope = negative_slope
    def forward(self, x2):
        t1 = self.conv(x2)
        v1 = t1 * 0.7422609026870728
        t2 = t1 > 0
        v2 = v1 * (self.negative_slope)
        t3 = t2 * (t1 - v1) + v1
        v3 = t3 * 0.9686799850463867
        t4 = t3 * 0.1901634699306488
        t5 = t3 + v3
        t6 = t4 < 0
        t7 = t4 > 0
        v4 = t5 * t6 + t7 * (v3 - t5)
        v5 = t5 * 0.9419570560455322
        t8 = t6 + 0.5
        v6 = t8 * 0.4625918221950531
        v7 = t5 * 0.3619235155582428
        t9 = t5 + v5
        t10 = t1 * 0.3142871146058083
        v8 = t9 * 0.4236503725767136
        v9 = t9 * 0.2088068524456024
        t11 = t3 + v7
        t12 = v4 * (v2 + v6 + v8 + t10 + v9 + t11)
        return v4
negative_slope = 0.3412144500732422
# Inputs to the model
x2 = torch.randn(1, 22, 109, 62)
