
class Block(torch.nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.features = torch.nn.ModuleList([torch.nn.ReLU(inplace=False), torch.nn.ConvTranspose2d(inp, out, 2, 2, 0, bias=True)])
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        result = concatenated_tensor
        for feature in self.features:
            result = feature(result)
        return (result, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
