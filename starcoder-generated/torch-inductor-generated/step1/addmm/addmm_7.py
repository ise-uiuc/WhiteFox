
class Model(torch.nn.Module):
    def __init__(self, h, w, d, out_ch):
        super().__init__()
        self.layer = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
 
	def forward(self, x, inp):
        v1 = self.layer(x)
        v2 = torch.flatten(v1, start_dim=1)
        v3 = v2.view(v2.size(0), -1, h, w)
        v4 = torch.sum(v3, dim=2)
        v5 = torch.sum(v4, dim=1)
        v6 = torch.flatten(v5, start_dim=1)
        v7 = v6 + inp
        v8 = v7.view(v7.size(0), v7.size(1), h, w)
        v9 = self.layer(v8)
        return v9

# Initializing the model
h, w, d = 64, 64, 256
m = Model(h, w, d, 10)

# Inputs to the model
x = torch.randn((1, 3, h, w))
inp = torch.randn((1, d))
