
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 4)
        self.sigmoid = torch.nn.Sigmoid()
        self.conv2d = torch.nn.Conv2d(4, 4, 3)
    def forward(self, x1):
        v1 = F.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1.transpose(2, 1)
        #v2 = v1.permute(0, 2, 1)
        v3 = v2.contiguous()
        v4 = self.sigmoid(v3)
        v5 = v4.unsqueeze(1)
        v6 = v5.contiguous()
        v7 = self.conv2d(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 2, 2)
