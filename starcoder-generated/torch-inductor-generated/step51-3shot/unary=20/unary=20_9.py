
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(30, 16, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2))
        self.conv_t2 = torch.nn.ConvTranspose2d(16, 10, kernel_size=(3, 3), stride=(1, 2), padding=(2, 2))
        self.conv_t3 = torch.nn.ConvTranspose2d(10, 5, kernel_size=(4, 4), stride=(1, 1), padding=(2, 2))
    def forward(self, x1):
        v1 = self.conv_t1(x1)
        v2 = F.relu(v1)
        v3 = self.conv_t2(v2)
        v4 = F.softmax(v3, dim=1)
        v5 = v4.transpose(1, 2)
        v6 = self.conv_t3(v5)
        v7 = F.sigmoid(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 30, 64, 64)
