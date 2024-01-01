
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
    def forward(self, input):
        x = self.conv(input)
        #x = torch.cat([x, x, x, x, x, x], dim=1)
        x = x.mean([2, 3])    
        x = x.unsqueeze(2)
        x = x.view(x.shape[0], x.shape[1], x.shape[2], 1, 1)
        x = x.repeat(1, 1, 1, 3, 3)
        x = x * x
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x * x
        return x
# Inputs to the model
input = torch.randn(1, 16, 64, 64)
