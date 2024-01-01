
class Model(nn.Module):
  def __init__(self):
    super().__init__()
    def f(x):
      return torch.nn.ReLU()(x) if x else F.pad(x, pad, 'constant', 0)
    features = [
      [
        torch.nn.Conv2d(3, 32, (3, 3), (1, 1)),
        [nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)],
        torch.nn.Conv2d(32, 64, (5, 5), (1, 1)),
        torch.nn.Sigmoid(),
        torch.nn.Conv2d(64, 512, (1, 1), (1, 1)),
        torch.nn.Conv2d(512, 425, (1, 1), (1, 1)), 
        [f(True), f(True),
        [f(True), F.pad(f(True), pad, 'constant', 0)],
      [
        torch.nn.Conv2d(350, 128, (1, 1), (1, 1)),
        torch.nn.Conv2d(128, 64, (1, 1), (1, 1)),
        torch.nn.Sigmoid(),
        [f(True), F.pad(f(True), pad, 'constant', 0)],
        [nn.ReLU(inplace=False), nn.ConvTranspose2d(64, 6, kernel_size=(1, 1), stride=(1, 1)) ]
        torch.nn.Flatten(),
      ]
    )
  ]
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
