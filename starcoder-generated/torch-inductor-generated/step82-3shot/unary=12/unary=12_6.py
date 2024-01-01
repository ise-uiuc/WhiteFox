
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1)
    def forward(self, input_tensor):
        t1 = self.conv(input_tensor)
        t2 = torch.nn.Sigmoid()
        t3 = torch.mul(t2(t1), t1)
        return t3
# Inputs to the model
input_tensor = torch.rand(1, 3, 64, 64)
