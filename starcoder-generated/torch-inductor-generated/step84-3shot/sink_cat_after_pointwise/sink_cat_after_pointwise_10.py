
class SinkCat(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.conv2 = torch.nn.Conv2d(3, 6, 5)
    def forward(self, x):
        x = torch.cat((self.conv1(x), self.conv2(x).mean(dim=-1, keepdims=True)), dim=1)
        x = torch.relu(x)
        return x.view(x.shape[0], -1)
# Inputs to the model
x = torch.randn(1, 3, 128, 128)
