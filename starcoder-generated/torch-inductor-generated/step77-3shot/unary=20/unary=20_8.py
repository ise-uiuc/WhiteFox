
class Model(torch.nn.Module):
    # 2-D convolutional layer with padding
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 2), bias=False)
        # ReLU nonlinear
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = self.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
