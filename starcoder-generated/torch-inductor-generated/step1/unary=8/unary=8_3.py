
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convt = torch.nn.ConvTranspose2d(8, 3, 3, stride=1, padding=1, output_padding=0)
 
    def forward(self, x):
        v1 = self.convt(x)
        v2 = v1 + 3
        v3 = torch.clamp(v2, 0, 6)
        v4 = v3 * 6
        return v4
    
# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 8, 7, 7)
