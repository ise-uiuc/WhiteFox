
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(196, 287, 2, input_padding=(1,1), padding=(1,1), dilation=1, groups=1, bias=True, padding_mode='zeros')
    def forward(self, x):
        t1 = self.conv(x)
        t2 = torch.tanh(t1)
        return t2
# Inputs to the model
x = torch.randn(1, 196, 32)
