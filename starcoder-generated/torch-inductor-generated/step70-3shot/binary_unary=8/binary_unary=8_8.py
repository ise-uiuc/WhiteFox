
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_ = torch.nn.Conv2d(3, out_in_out_dim, 3, padding=0)
 
    def forward(self,x1):
        v = self.conv_(x1)
        return v

x1 = torch.randn(batch_size, in_in_out_dim, 224, 224)
model.eval()
model(x1)