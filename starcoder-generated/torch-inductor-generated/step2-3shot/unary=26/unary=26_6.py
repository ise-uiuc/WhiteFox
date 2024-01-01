
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose2d = torch.nn.ConvTranspose1d(1, 20, kernel_size=1)
    def forward(self, tensor):
        tensor = self.conv_transpose2d(tensor)
        mask = tensor > 0
        result = torch.where(mask, tensor * 2, tensor / 2)  # tensor should be float, otherwise: "RuntimeError: Output type of  is incompatible with input type of "
        return result
# Inputs to the model
x = torch.randn(1, 1, 20)
negative_slope = 0.1
