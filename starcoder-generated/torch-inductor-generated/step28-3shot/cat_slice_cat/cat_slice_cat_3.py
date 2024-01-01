
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x, y):
        v0 = torch.cat([x, y], dim=1)
        v1 = v0[:, 0:9223372036854775807]
        v2 = v1[:, 0:v0.size(1)]
        v3 = torch.cat([v0, v2], dim=1)
        return v3

# Initializing the model
m = Model().to(device=device)

# Inputs to the model
x = torch.randn(7, 8, 3, 224, 224).to(device=device)
y = torch.randn(7, 6, 3, 224, 224).to(device=device)
