
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 2, 1, stride=1, padding=1)
    def forward(self, input_tensor):
        t1 = self.conv(input_tensor)
        t2 = t1 + 3
        t3 = torch.clamp(t2, 0, 6)
        t5 = t3 * t1
        t6 = torch.mm(t5, t3)
        t7 = torch.div(t6, 6)
        return t7
# Inputs to the model
input_tensor = torch.randn(1, 2, 3, 3, requires_grad=True)
