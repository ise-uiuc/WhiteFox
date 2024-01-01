
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super(ModelTanh, self).__init__()
        self.conv = torch.nn.Conv2d(3, 3, (1,1), stride=(1,1))
        self.tanh = torch.nn.Tanh()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.tanh(x)
        return x
# Inputs to the model
input = torch.randn(1, 3, 224, 224)
