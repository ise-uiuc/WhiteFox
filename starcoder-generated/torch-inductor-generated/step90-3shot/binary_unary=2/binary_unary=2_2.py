
def get_block(c_in, c_out):
    return torch.nn.Sequential(
    torch.nn.Conv2d(c_in, c_out, 3, stride=2, padding=1),
    torch.nn.ReLU(),
    torch.nn.Conv2d(c_out, c_out, 3, stride=2, padding=1),
    torch.nn.ReLU(),
    torch.nn.Conv2d(c_out, c_out, 3, stride=2, padding=1),
    torch.nn.ReLU()
)

def get_layers(c_in, c_out):
    blocks = []
    for i in range(4):
        c_in = c_out * (2**i)
        c_out = c_out * (2**i) * (2**i) if i > 0 else c_out
        blocks.append(get_block(c_in, c_out))
    return torch.nn.Sequential(*blocks)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = get_layers(3, 4)
    def forward(self, x1):
        v1 = self.layers(x1)
        v2 = v1 - 20
        v3 = F.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
