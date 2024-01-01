
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose1d(333, 5, kernel_size=3, stride=1, padding=2, bias=True)
    def forward(self, x349):
        v1 = self.conv_t(x349)
        v2 = v1 >= 0
        v3 = v1 * torch.max(torch.FloatTensor(x349.shape[0]), torch.FloatTensor([torch.mean(torch.abs(x349))]))
        v4 = torch.where(v2, v1, v3)
        return v4 + torch.nn.functional.adaptive_avg_pool1d(v4, (1))
# Inputs to the model
x349 = torch.randn(19, 333, 39)
