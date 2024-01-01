
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(1, 64, 7, stride=1, padding=3, output_padding=1, bias=False) 
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return torch.split(concatenated_tensor, [1, 1, 1], dim=1)
# Inputs to the model
x1 = torch.randn(1, 9, 9, 3)
