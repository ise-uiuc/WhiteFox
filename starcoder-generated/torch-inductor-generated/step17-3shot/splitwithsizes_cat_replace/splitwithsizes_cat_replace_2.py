
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 32, 5, 1, 2), torch.nn.Conv2d(32, 32, 3, 2, 1))
        self.cat = torch.nn.Sequential(torch.nn.Sigmoid())
    def forward(self, x0):
        split_tensor = torch.split(x0, 6, dim=3)
        split_tensors = [torch.stack(tensors, dim=0) for tensors in zip(*split_tensor)]
        concatenated_tensor = torch.cat(split_tensors, 3)
        return (concatenated_tensor, torch.split(x0, 6, dim=3))
# Inputs to the model
x0 = torch.randn(3, 64, 1, 128)
