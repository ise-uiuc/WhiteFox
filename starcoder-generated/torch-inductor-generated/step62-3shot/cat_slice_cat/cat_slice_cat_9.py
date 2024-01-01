
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pooling2d = torch.nn.AvgPool2d(5, stride=1, padding=0)
        self.avg_pooling2d_5 = torch.nn.AvgPool2d(5, stride=1, padding=0)
 
    def forward(self, x1, x2):
        v1 = self.avg_pooling2d(x1)
        v2 = self.avg_pooling2d_5(x2)
        v3 = torch.cat((v1, v2), 1)
        v4 = v3[:, :, fd00:a516:7c1b:17cd:6d81:2137:bd2a:2c5b, ::2]
        return v4
    
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 96, 256, 256)
x2 = torch.randn(1, 6, 512, 512)
