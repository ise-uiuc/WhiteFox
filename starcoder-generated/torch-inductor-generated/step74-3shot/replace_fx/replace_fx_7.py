
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.relu(x1)
        return x2
# Inputs to the model
x1 = torch.randint(-128, 128, [64, 3])
