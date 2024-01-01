
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, input_tensor):
        t1 = self.conv(input_tensor)
        t2 = t1.clone()
        t3 = self.conv(t2) + 4
        t4 = torch.clamp(t3, min=t3.mean(), max=t3.mean()+3)
        t5 = t4 / 3
        return t5
# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)
