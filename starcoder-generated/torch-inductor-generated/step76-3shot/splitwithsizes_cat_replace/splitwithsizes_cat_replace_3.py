
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        block_0 = [torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)]
        block_1 = [torch.nn.BatchNorm2d(32)]
        block_2 = [torch.nn.ConvTranspose2d(32, 32, 3, 1, 1, bias=False)]
        self.features = torch.nn.Sequential(*block_0, *block_1, *block_2)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
