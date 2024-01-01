
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super(ModelTanh, self).__init__()
        self.a_1 = torch.nn.Conv2d(3, 3, kernel_size=(1, 1), stride=(1, 1), padding=(0,0), dilation=(1, 1), groups=1) 
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        x = self.a_1(x)
        v2 = torch.tanh(x)
        return v2
# Inputs to the model
x = torch.randn(3, 3, 3, 3)
