
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        split_tensors = torch.split(v6, split_sizes=8, dim=1)
        concatenated_tensor = torch.cat([split_tensors[7], split_tensors[1], split_tensors[0], split_tensors[6], split_tensors[3], split_tensors[2], split_tensors[5], split_tensors[4]], dim=1)
        return torch.sum(concatenated_tensor)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
