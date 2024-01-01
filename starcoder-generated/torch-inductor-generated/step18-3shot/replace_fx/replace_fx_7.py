
from torch.nn.modules import dropout
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.drop1 = dropout.Dropout(0.4)
        self.conv1 = torch.nn.Conv1d(4, 8, 4, bias=False)
        self.conv2 = torch.nn.Conv1d(8, 4, 4, bias=False)
   
    def forward(self, data):
        x = self.conv1(data)
        x = self.drop1(x)
        x = self.conv2(x)
        out = torch.sum(x)
        #out = torch.mul(out, 100) # Uncomment to prevent optimization
        return out
# Inputs to the model
data1 = torch.randn(1, 4, 4)
