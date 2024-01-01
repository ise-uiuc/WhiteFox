
class Model(torch.nn.Module):
    def __init__(self, p1):
        super().__init__()
        self.p1 = p1
        self.conv = torch.nn.Conv1d(in_channels=2, out_channels=2, kernel_size=2)
    def forward(self, x1):
        x2 = self.conv(x1) ** self.p1
        x3 = torch.nn.functional.dropout(x2)
        x4 = torch.rand_like(x3)
        return x4
p1 = 1
# Inputs to the model
x1 = torch.randn(1, 2, 2)
