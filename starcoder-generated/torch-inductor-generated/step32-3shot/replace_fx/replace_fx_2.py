
class model(torch.nn.Module):
    def __init__(self):    
        super().__init__()    
        self.conv = torch.nn.Conv3d(32, 12, 2, bias=False)
        self.dropout_fn = torch.nn.Dropout2d(0.5, inplace=False)
    def forward(self, input, weight, bias):
        x = input.permute(0, 2, 3, 4, 1)
        y = self.conv(x)
        y = y + self.conv.weight
        z = self.dropout_fn(y)
        x = z.permute(0, 4, 1, 2, 3)
        x = input.view(input.shape)
        return input, x, weight, bias
# Inputs to the model
input = torch.randn(1, 32, 2, 2, 2)
weight = torch.randn(1024, 16384)
bias = torch.randn(1024)
