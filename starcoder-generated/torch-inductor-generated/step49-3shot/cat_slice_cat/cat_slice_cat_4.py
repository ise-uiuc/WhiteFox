
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, v1_1, v1_2):
        v1_3 = torch.cat([v1_1, v1_2], dim=1)
        v1_5 = v1_3[:, 1:size]
        v1_6 = torch.cat([v1_3, v1_5], dim=1)
        return v1_6

# Initializing the model
m = Model()

# Input tensors to the model
v1_1 = torch.randn(1, 3, 64, 64)
v1_2 = torch.randn(1, 3, 64, 64)
