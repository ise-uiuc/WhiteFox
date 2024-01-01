
class Model(torch.nn.Module):
    def __init__(self):
        self.layer1 = torch.nn.Conv2d(3, 32, 5)
    def forward(self, x):
        x = self.layer1(x)
        x = torch.nn.functional.dropout(x, p=0.5)
        return x
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
