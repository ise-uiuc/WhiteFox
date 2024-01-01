
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5, bias=True)
        self.conv = torch.nn.Conv2d(6, 1, kernel_size=(5,), stride=(1,), padding=(2,), groups=(1,))
    def forward(self, x):
        v1 = self.linear(x)
        v1 = v1.reshape(2, 1, 3)
        v2 = torch.clamp_min(v1, 0)
        v3 = torch.clamp_max(v2, 6)
        v4 = v3.reshape(3, 1, 1)
        v5 = v4 * 4
        v6 = torch.transpose(v5, 1, 2)
        return self.conv(v6)
# Inputs to the model
x = torch.randn(2, 3)
