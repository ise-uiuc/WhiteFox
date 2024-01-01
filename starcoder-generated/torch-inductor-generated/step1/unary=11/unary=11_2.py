
class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(3, 8, 3, stride=1, padding=0)
 
    def forward(self, x):
        v1 = self.conv_t(x)
        v2 = torch.clamp_min(v1, 0)
        v3 = torch.clamp_max(v2, 6)
        v4 = v3 + 3
        return v4 / 6
    
# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
