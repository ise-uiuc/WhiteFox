
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.module_1 = torch.nn.ModuleDict({
            "1": torch.nn.Softplus(),
            "2": torch.nn.LayerNorm(4, eps=0.4594111),
            "3": torch.nn.Softplus(),
            "4": torch.nn.BatchNorm2d(1, affine=True),
            "5": torch.nn.MaxPool2d(3)
        })
    def forward(self, x1):
        v1 = self.module_1["1"](x1)
        v2 = self.module_1["2"](x1)
        v4 = self.module_1["3"](v1)
        v5 = self.module_1["4"](x1)
        v6 = self.module_1["5"](v5)
        v7 = self.module_1["4"].weight
        v8 = self.module_1["4"].bias
        v9 = self.module_1["4"].running_mean
        v10 = self.module_1["4"].running_var
        v11 = self.module_1["4"].num_batches_tracked
        return v9
# Inputs to the model
x1 = torch.randn(1, 1, 3, 3)
x2 = torch.randn(1, 1, 1, 1)
x3 = torch.randn(1, 1, 5, 5)
x4 = torch.randn(1, 1, 7, 7)
