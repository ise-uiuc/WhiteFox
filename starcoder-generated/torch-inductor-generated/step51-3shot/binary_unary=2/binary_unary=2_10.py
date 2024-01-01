
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(91, 225, (7, 7), stride=(1, 1), padding=(3, 3))
        self.conv2 = torch.nn.Conv2d(225, 512, (1, 1), stride=(1, 1), padding=(0, 0))
    def forward(self, input):
        x1 = self.conv1(input)
        x2 = F.max_pool2d(x1, kernel_size=3, stride=2, padding=1, ceil_mode=False)
        x3 = x2 - 1.7321
        x4 = F.relu(x3)
        x5 = self.conv2(x4)
        x6 = F.local_response_norm(x5, size=5, alpha=0.0001, beta=0.75, k=1.0)
        return x6
# Inputs to the model
input = torch.randn(1, 91, 129, 64) 
