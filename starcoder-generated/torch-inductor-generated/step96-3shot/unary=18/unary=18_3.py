
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # 256x448 input image
        self.Conv_1 = nn.Conv2d(3, 32, 2, stride=1, padding=1)

        # (32,32) output for max pooling
        self.pool = nn.MaxPool2d(8,1)

        # (16,16) output size from convolution
        self.Conv_2 = nn.Conv2d(32, 64, 8, stride=8)

        # (9,9) output for max pooling
        self.pool2 = nn.MaxPool2d(2,2)

        # (1,1) output size from convolution
        self.Conv_3 = nn.Conv2d(64,32,4,stride=4)

        # (1,1) output size from conv
        self.Conv_4 = nn.Conv2d(32,1,7,stride=1)
    def forward(self,x_t):
        t3 = self.Conv_1(x_t)
        t4 = nn.Sigmoid()(t3)
        v1 = self.pool(t4)
        v1 = self.Conv_2(v1)
        v1 = nn.ReLU()(v1)
        v1 = self.pool2(v1)
        v1 = self.Conv_3(v1)
        v1 = nn.Tanh()(v1)
        v1 = self.Conv_4(v1)
        x = nn.Sigmoid()(v1)
        return x
# Inputs to the model
x1 = torch.randn(1, 3, 256, 448)
