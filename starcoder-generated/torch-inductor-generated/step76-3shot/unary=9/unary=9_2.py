
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = torch.add(t1, torch.tensor([3.], dtype=torch.float))
        t3 = torch.clamp(t2, min=0, max=6)
        t4 = torch.true_divide(t2, torch.tensor([6.], dtype=torch.float))            
        return t4
# Inputs to the model
input_data = torch.randn(1, 3, 64, 64)
