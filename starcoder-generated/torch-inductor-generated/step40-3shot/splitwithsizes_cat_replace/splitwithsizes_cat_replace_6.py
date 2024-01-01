
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.ReLU(inplace=True), torch.nn.ConvTranspose2d(4, 8, 5, 5, 0, 0, bias=False))
    def forward(self, v1):
        split_tensors = torch.split(v1, (1, 1, 1), dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, (1, 1, 1), dim=1))
# Inputs to the model
x1 = torch.randn(1, 4, 3, 3)
