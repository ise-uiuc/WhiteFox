
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convtranspo1 = nn.ConvTranspose2d(3, 16, 7, stride=4, padding=3)
        self.leaky_relu_1 = nn.LeakyReLU(0.2, inplace=True)
        self.convtranspo2 = nn.ConvTranspose2d(16, 1, 7, stride=4, padding=3)
        #self.batch_normalization_1 = nn.BatchNorm2d(1)
        self.log_softmax = nn.LogSoftmax(dim=1)
    def forward(self, x1):
        v1 = self.convtranspo1(x1)
        v2 = self.leaky_relu_1(v1)
        v3 = self.convtranspo2(v2)
        #v4 = torch.sigmoid(v3)
        v4 = self.log_softmax(v3)
        return v4
# Inputs to the model
x1 = torch.Tensor([1, 3, 16, 16])
x1 = x1.view(1, 3, 4, 4)
