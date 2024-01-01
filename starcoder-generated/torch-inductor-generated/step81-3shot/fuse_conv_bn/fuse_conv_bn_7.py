
# Use `torch.nn.Linear()`, `torch.nn.LayerNorm()`, and `torch.nn.ConvTranspose()` etc. when creating models.
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(113)
        self.conv3 = torch.nn.Sequential(torch.nn.Conv2d(10, 5, 4))
        self.conv1 = torch.nn.Sequential(torch.nn.Linear(10, 5))
        torch.manual_seed(453)
        self.conv4 = torch.nn.Conv2d(5, 1, 1)
        torch.manual_seed(51)
        self.layernorm4 = torch.nn.LayerNorm((6,5), 3.1)
        self.batchnorm = torch.nn.BatchNorm2d(1)
        self.batchnorm1 = torch.nn.BatchNorm2d(5)
    def forward(self, x3, t3):
        # For `torch.nn.LayerNorm()`, only 2D input is allowed. Hence we use `torch.unsqueeze()` to add a new dimension.
        t3 = torch.nn.functional.interpolate(torch.unsqueeze(t3, 1), [3, 4])
        t2 = self.conv1(x3)
        x3 = t3
        x2 = t2
        x1 = t1
        # For `torch.nn.ConvTranspose()`, only 4D input is allowed. Hence we use `torch.unsqueeze()` to add new dimensions.
        s3 = self.conv4(torch.nn.functional.interpolate(torch.unsqueeze(x3, 2), [5, 5]))
        x3 = torch.cat([x3, s3], 1)
        z4 = self.layernorm4(x3 + s3)
        x = self.batchnorm(z4)
        return x, 
# Inputs to the model
x3 = torch.randn(1, 10, 10, 10)
t3 = torch.randn(1, 3, 4)
