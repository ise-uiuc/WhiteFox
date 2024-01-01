
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 64, 1)
    def forward(self, x):
        tanh_out = torch.tanh(self.conv(x))
        tanh_out = torch.abs(tanh_out)
        tanh_out = torch.tanh(tanh_out)
        tanh_out_1 = torch.abs(tanh_out)
        tanh_out_2 = torch.tanh(tanh_out_1)
        tanh_out_3 = torch.sigmoid(tanh_out_2)
        return tanh_out_3
# Inputs to the model
x = torch.randn(1, 1, 100, 100)
