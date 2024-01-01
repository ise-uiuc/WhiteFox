
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d((17, 20), 13, (1, 6), stride=(1, 2), bias=True)
        self.conv_t2 = torch.nn.ConvTranspose2d(17, 9, (1, 7), stride=(1, 1), bias=False)
        self.conv_t3 = torch.nn.ConvTranspose2d((41, 60), 11, (5, 7), stride=(3, 4), bias=False, output_padding=(10, 9))
    def forward(self, x1):
        x = torch.sigmoid(self.conv_t1(x1))
        x = torch.sigmoid(self.conv_t2(x))
        x = torch.sigmoid(self.conv_t3(x))
        return x
# Inputs to the model
x1 = torch.randn(2, 17, 51, 53)
