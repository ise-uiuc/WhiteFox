
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x1):
        v1 = self.conv(x1)
        v11 = v1.sum(axis=-1, keepdims=False).sum(axis=-2, keepdims=False)
        v12 = v11 + 1
        v13 = v12.mean(axis=1).mean(axis=1)
        v21 = v1.unsqueeze(1)
        v22 = v21.permute((0, 2, 1, 3))
        v23 = v22 - 1.0
        v24 = v23.permute((0, 2, 1, 3))
        v31 = v11.unsqueeze(1)
        v32 = v31.permute((0, 2, 1, 3))
        v33 = v32 + 3.0
        v34 = v33.permute((0, 2, 1, 3))
        v41 = v13.unsqueeze(1)
        v42 = v41.permute((0, 2, 1, 3))
        v43 = v42 * 7.0
        v44 = v43.permute((0, 2, 1, 3))
        v5 = v24 + v34 + v44
        return v5

# Initializing the model
__m__ = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
