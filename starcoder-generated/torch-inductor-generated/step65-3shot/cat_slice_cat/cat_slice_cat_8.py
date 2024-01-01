
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t1 = torch.nn.Conv2d(3, 16, (5, 5), stride=(1, 1), padding=(1, 1))
        self.t2 = torch.nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), dilation=1, ceil_mode=False)
        self.t3 = torch.nn.Conv2d(16, 32, (3, 3), stride=(1, 1), padding=(1, 1))
        self.t4 = torch.nn.AdaptiveAvgPool2d(output_size=[100, 100])
        self.t5 = torch.nn.Conv2d(32, 64, (3, 3), stride=(1, 1), padding=(1, 1))
        self.t6 = torch.nn.AdaptiveAvgPool2d(output_size=[100, 100])
        self.t7 = torch.nn.Conv2d(64, 128, (3, 3), stride=(1, 1), padding=(1, 1))
        self.t8 = torch.nn.AdaptiveAvgPool2d(output_size=[100, 100])
        self.t9 = torch.nn.Conv2d(128, 64, (3, 3), stride=(1, 1), padding=(1, 1))
        self.t10 = torch.nn.AdaptiveAvgPool2d(output_size=[100, 100])
        self.t11 = torch.nn.Flatten()
        self.t12 = torch.nn.Linear(6400, 1)

    def forward(self, *xs):
        t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12 = self.t1, self.t2, self.t3, self.t4, self.t5, self.t6, self.t7, self.t8, self.t9, self.t10, self.t11, self.t12
        t1 = t1(xs[63])
        t2 = t2(t1)
        t3 = t3(xs[31])
        t4 = torch.cat([t2, t3], dim=1)
        t5 = t5(t4)
        t6 = t6(t5)
        t7 = t7(t6)
        t8 = t8(t7)
        t9 = t9(t8)
        t10 = t10(t9)
        t11 = t11(t10)
        t12 = t12(t11)
        return t12

# Initializing the model
m = Model()

# Inputs to the model
xs = torch.randn(100, 3, 224, 224)
