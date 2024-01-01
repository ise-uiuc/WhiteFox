
class Model(torch.nn.Module):
    def forward(self, x1):
        v1 = torch.nn.functional.conv_transpose2d(x1, None, kernel_size=(9, 9), stride=(2, 2), padding=(1, 1))
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 16, 14, 14)
