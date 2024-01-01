
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose2d1 = torch.nn.ConvTranspose2d(in_channels=2, out_channels=3, kernel_size=(3,), stride=(2,),
                                                            padding=(1,))
        self.batch_norm2d1 = torch.nn.BatchNorm2d(num_features=3)
        self.linear3 = torch.nn.Linear(in_features=9, out_features=20)
    def forward(self, x0, x1):
        x0 = self.conv_transpose2d1(x0)
        x1 = self.batch_norm2d1(x1)
        y0 = x1.transpose()
        y1 = x0.view(x0.size())
        y2 = y1 + x1
        y3 = y1.neg()
        y4 = y0 + y3
        y5 = y0 + y2
        y6 = y4 + x1.transpose()
        y7 = x1 + y6
        y8 = self.linear3(y5)
        return y5, y7, y8
# Inputs to the model for trace
input1 = torch.randn(4, 2, 8, 8)
input2 = torch.randn(4, 3, 4, 4)
