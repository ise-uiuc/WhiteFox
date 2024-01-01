
class Model(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.slice1 = torch.tensor([9223372036854775807 for _ in range(size)])
 
    def forward(self, x1, x2):
        y1 = torch.cat((x1, x2), dim=1)
        y2 = y1[:, 0:9223372036854775807]
        y3 = y2[:, 0:self.slice1.size()]
        y4 = torch.cat((y3, y2), dim=1)
        return y4

# Initializing the model
m = Model(2)

# Inputs to the model
x1 = torch.randn(3, 2, 4, 5)
x2 = torch.randn(3, 1, 4, 5)
