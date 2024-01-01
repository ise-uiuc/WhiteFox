
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.conv2d = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=(0,), dilation=(1,))
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v2 = torch.reshape(v2, (1, 1, 2, 2))
        v2 = v2.permute(0, 3, 2, 1)
        v2 = self.conv2d(v2)  # NCHW input with conv2d that operates on NHWC
        v2 = v2.permute(0, 3, 2, 1)
        v2 = torch.sum(v2, dim=(1, 2))  # sum each (2) channel along C dimension, then NHWC will be flattend as CHW and then 1-D tensor.
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
