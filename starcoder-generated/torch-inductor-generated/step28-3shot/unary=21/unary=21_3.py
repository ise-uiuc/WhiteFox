
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=[12, 56], stride=[12, 56])
    def forward(self, t0):
        t1 = self.conv2d(t0)
        t2 = torch.tanh(t1)
        return t2
# Inputs to the model
t0 = torch.randn(1, 1, 12, 56, requires_grad=True)
