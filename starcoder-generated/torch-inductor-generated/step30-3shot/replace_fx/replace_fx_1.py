        
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.modules.resnet.Bottleneck(16, 64, 2)
    def forward(self, x1):
        v = torch.nn.functional.dropout(x1, p=0, training=False)
        x = self.layer(v)
        y = torch.matmul(x, x)
        return y
# Inputs to the model
x1 = torch.randn(1, 16, 4, 4)
