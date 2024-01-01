
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.view = torch.nn.Sequential(
            view(),
            torch.nn.Linear(in_features=100, out_features=1, bias=True)
        )
    def forward(self, x1):
        v1 = self.avg_pool(x1)
        v2 = v1.view(v1.size(0), -1)
        v3 = self.view(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 100, 14, 14)
