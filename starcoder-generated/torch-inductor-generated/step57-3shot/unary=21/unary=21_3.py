
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(256,256,128,1,0,1,16)
        self.tanh = torch.nn.Tanh()  
    def forward(self,x):
        v1 = self.conv(x)
        v2 = self.tanh(v1)
        return v2
# Inputs to the model
x = torch.randn(1, 256, 569200)
