
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = torch.nn.ModuleList([
            torch.nn.Conv1d(256, 128, 9, stride=1) for _  in range(4)
        ])
    def forward(self, x1):
        v1 = x1
        for m in self.conv_block:
            v1 = m(v1)
        return v1.permute(2, 1, 0).squeeze(-1)
# Inputs to the model
x1 = torch.randn(300, 16, 256)
