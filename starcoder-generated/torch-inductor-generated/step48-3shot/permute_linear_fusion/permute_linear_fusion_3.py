
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.dropout = torch.nn.Dropout(0.5)
        self.sigmoid = torch.nn.Identity()
        self.maxpool = torch.nn.MaxPool1d(kernel_size=2, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.adaptive_avg_pool1d = torch.nn.AdaptiveAvgPool1d((1,))
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = self.sigmoid(v2)
        v4 = self.dropout(v3)
        v5 = v4.unsqueeze(1)
        v6 = self.maxpool(v5)
        v7 = v6.squeeze(1)
        v8 = self.adaptive_avg_pool1d(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 2, 2)
