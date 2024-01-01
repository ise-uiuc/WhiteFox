
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 =   torch.nn.Conv2d(2, 3, 3, stride=2, padding=3, dilation=1, groups=1, bias=True, padding_mode='zeros')
    def forward(self, x):
        a1 = torch.nn.functional.dropout(self.conv2.weight)
        a2 = torch.rand_like(self.conv2.weight)
        a3 = torch.rand_like(self.conv2.weight, dtype=torch.float)
        a4 = torch.rand_like(self.conv2.weight, dtype=torch.float)
        a5 = torch.rand_like(self.conv2.weight, dtype=torch.float)
        a6 = torch.nn.functional.dropout(self.conv2.weight)
        return torch.nn.functional.dropout(a5)
# Inputs to the model
x1 = torch.randn(1, 2, 4, 4)
