
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super(ModelTanh, self).__init__()
        self.conv = torch.nn.Conv1d(20, 20, 20, stride=2, padding=5, padding_mode='circular')
    def forward(self, x3):
        e1 = self.conv(x3)
        e2 = torch.tanh(e1)
        return e2
# Inputs to the model
x3 = torch.randn(10, 20, 20)

