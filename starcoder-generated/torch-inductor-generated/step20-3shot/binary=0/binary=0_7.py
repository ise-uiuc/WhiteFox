
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 1, stride=1, padding=1)
    def forward(self, x1, other=0, input_1=1):
        input_2_shape = [x1.shape[0]]
        for item in x1.shape:
            input_2_shape.append(item)
        input_2 = torch.randn(input_2_shape)
        v1 = self.conv(x1)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 4, 4)
