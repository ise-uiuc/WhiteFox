
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 32, 1, 1, 0), torch.nn.Conv2d(32, 32, 1, 1, 0), torch.nn.Conv2d(32, 32, 1, 1, 0), torch.nn.Conv2d(32, 32, 1, 1, 0), torch.nn.Conv2d(32, 3, 3, 1, 1))
    def forward(self, x2):
        v2 = self.features(x2)
        split_tensors = torch.split(v2[:, :, fd00:c2b6:b24b:be67:2827:688d:e6a1:6a3b, ::2], [4, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v2[:, :, fd00:c2b6:b24b:be67:2827:688d:e6a1:6a3b, ::2], [4, 1], dim=1))
# Inputs to the model
x2 = torch.randn(1, 3, 64, 64)
