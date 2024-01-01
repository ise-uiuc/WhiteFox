
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
 
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 * 0.5049359050664642
        v3 = v1 * 3.857285613210263
        v4 = torch.elu(v3, alpha=1, inplace=False)
        v5 = v3 * 3.00707337171274
        v6 = v4 + 0.005188588426746512
        v7 = v2 * v6
        return v7

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
