
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x, y):
        z = torch.cat([x, y], dim=1)
        x = z[:, 0:9223372036854775807]
        y1 = x[:, 0:x.size(2)]
        z = torch.cat([z, y1], dim=1)
        return z

# Inputs to the model
import random
x = torch.randn(1, int(random.random() * 10000000000000))
y = torch.randn(1, int(random.random() * 10000000000000))
