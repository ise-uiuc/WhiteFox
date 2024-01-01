
t = torch.tensor([[[0, 255], [0, 255]], [[255, 0], [255, 0]]], dtype=torch.uint8)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 1, 1, stride=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - t / 255
        return v2
# Inputs to the model
x = torch.zeros([1, 2, 2, 2], dtype=torch.uint8)
