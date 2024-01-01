
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.Conv1 = torch.nn.Conv2d(3, 4, 3)
        self.Conv2 = torch.nn.Conv2d(1, 2, 1)
        self.Conv3 = torch.nn.Conv2d(4, 1, 1)
    def forward(self, x1):
        v1 = self.Conv1(x1)
        v2 = torch.sigmoid(v1)
        
        v3 = self.Conv2(v2)
        v4 = torch.sigmoid(v3)
        
        v5 = self.Conv3(v4)
        v6 = torch.sigmoid(v5)
        
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
