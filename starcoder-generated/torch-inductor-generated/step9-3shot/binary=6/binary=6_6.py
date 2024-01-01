
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x1, x2):
        v1 = torch.nn.functional.linear(x1, x2)
        v2 = v1 - x2
        return v2

# Initializing the model
batch_size = 10
m = Model()
x1 = torch.randn(batch_size, 3, 64, 64)
x2 = torch.randn(3, 3, 1, 1)
