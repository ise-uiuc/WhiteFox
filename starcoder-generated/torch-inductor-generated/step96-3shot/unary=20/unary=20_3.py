
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = ConvTranspose2d_1(x1, out_channels=1, kernel_size=[7, 7], stride=[7, 7])
        v2 = sigmoid(v1)
        return v2
# Inputs to the model
Input_1 = torch.randn(1, 912, 1, 88)
