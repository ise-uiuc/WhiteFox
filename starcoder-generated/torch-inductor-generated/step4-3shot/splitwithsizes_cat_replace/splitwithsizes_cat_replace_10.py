
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x):
        x = self.conv(x)
        split_tensors = torch.split(x, [5, 7, 3, 7, 1, 7], dim=1)
        concatenated_tensor = torch.cat([split_tensors[i] for i in range(len(split_tensors))], dim=1)
        v1 = (x - concatenated_tensor).sum()
        v2 = torch.softmax(x, dim=1).sum()
        return v1 + v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 8, 8)
