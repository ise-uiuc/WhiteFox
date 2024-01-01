
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass
 
    def forward(self, img, fushion):
        out1 = torch.matmul(img, fushion)
        out2 = torch.matmul(img, fushion)
        out3 = torch.matmul(img, fushion)
        fc = torch.cat([out1, out2, out3], dim=1)
        return fc

# Initializing the model
m = Model()

# Inputs to the model.
img = torch.randn(1, 104, 400)
fushion = torch.randn(104, 256)
