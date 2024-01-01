
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x1):
        splits = torch.split(x1, [2, 2, 2, 2, 2, 2, 2, 2], dim=2)
        concatenated = torch.cat(splits, dim=2)
        return torch.sum(concatenated**2)

# Initializing the model
m = Model()

# Inputs to the model
batch_size = 1
x1 = torch.randn(batch_size, 3, 64, 64)
