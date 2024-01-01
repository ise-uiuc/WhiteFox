
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 512 -> 256
        self.linear1 = torch.nn.Linear(512, 256, bias=False)
        # 256 -> 1
        self.linear3 = torch.nn.Linear(256, 1, bias=False)
 
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = F.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.linear3(v3)
        v5 = v4.squeeze()
        return v5

x1 = torch.randn(64, 512)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(64, 512)
