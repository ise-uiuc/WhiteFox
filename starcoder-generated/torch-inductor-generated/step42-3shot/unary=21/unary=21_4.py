
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super(ModelTanh, self).__init__()
        self._conv1 = torch.nn.Conv2d(1,4,5)
        self._tanh = torch.nn.Tanh()
    def forward(self, input):
        x1 = self._conv1(input)
        y1 = self._tanh(x1)
        return y1
# Inputs to the model
input = torch.randn(1,1,10,10)
# model ends
