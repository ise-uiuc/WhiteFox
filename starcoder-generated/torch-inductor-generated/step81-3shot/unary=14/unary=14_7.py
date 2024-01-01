
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_2 = torch.nn.Conv2d(1, 1, 53, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv_2(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = torch.nn.functional.interpolate(v3, scale_factor=[0.7471], recompute_scale_factor=True, mode='nearest')
        v5 = torch.zeros(list(v4.size()), dtype=torch.int32)
        v6 = torch.nn.functional.embedding(v5, x1, padding_idx=0, max_norm=3.9863237848191805, norm_type=2.0, scale_grad_by_freq=True)
        v7 = torch.nn.functional.pad(v6, (6, 4, 1, 7), mode='constant', value=1)
        v8 = torch.nn.functional.celu(v7)
        v9 = torch.nn.functional.linear(x1, x1)
        v10 = torch.nn.functional.nll_loss(x1, v8)
        v11 = torch.nn.functional.pad(x1, (21, 68, 0, 4), mode='constant', value=7.8196870193481445)
        return v9
# Inputs to the model
x1 = torch.randn(1, 1, 43, 29)
x2 = torch.randn(1, 1, 43, 29)
