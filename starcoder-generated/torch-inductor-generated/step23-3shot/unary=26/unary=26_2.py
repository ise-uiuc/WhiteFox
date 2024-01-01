
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t0 = torch.nn.ConvTranspose1d(1, 1, 3, stride=1)
        self.conv_t1 = torch.nn.ConvTranspose1d(1, 1, 3, stride=1)
    def forward(self, input_tensor):
        y = self.conv_t0(input_tensor)
        _mask = y > 0
        _tensor = torch.where(_mask, y, 0.7*y)
        return self.conv_t1(_tensor)+_tensor
# Inputs to the model
input_tensor = torch.randn(1,1,3)
