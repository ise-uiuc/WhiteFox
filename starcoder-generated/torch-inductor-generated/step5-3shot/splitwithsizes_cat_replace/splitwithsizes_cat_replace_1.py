
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.split(v1, [1, 4, 2, 3, 1], dim=1)
        concat_list = [v2[i] for i in range(len(v2))]
        v3 = torch.cat(concat_list, dim=1)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
