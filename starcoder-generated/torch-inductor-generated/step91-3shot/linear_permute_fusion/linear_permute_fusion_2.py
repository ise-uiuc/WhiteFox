
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.max_pooling = torch.nn.MaxPool2d(4)
    def forward(self, x1):
        v1 = F.adaptive_avg_pool2d(x1, self.max_pooling.output_size)
        v2 = v1.transpose(0, 3)
        v3 = v2.transpose(0, 1)
        return v3
# Inputs to the model
x3 = torch.randn(1, 2, 2)
