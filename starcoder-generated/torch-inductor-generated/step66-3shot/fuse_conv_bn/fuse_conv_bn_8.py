
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1, stride=1, padding=1, groups=1, bias=True)     
    def forward(self, x):
        x1 = self.conv(x)
        x2 = torch.nn.functional.batch_norm(x1, torch.Tensor([1,]), torch.Tensor([1,]), [1,], torch.Tensor([0,]), torch.Tensor([0,]))
        return x2
# Inputs to the model
x = torch.randn(1, 16, 14, 14)
