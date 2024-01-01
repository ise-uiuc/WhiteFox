
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.module_1 = torch.nn.ModuleList([torch.nn.ConvTranspose2d(4, 6, kernel_size=1, stride=1, padding=1), torch.nn.ConvTranspose2d(4, 8, kernel_size=2, stride=2, output_padding=1)])
    def forward(self, x1):
        v1 = self.module_1[0](x1)
        v2 = self.module_1[1](x1)
        v4 = self.module_1[0].weight
        v5 = self.module_1[0].bias
        v6 = self.module_1[1].weight
        v7 = self.module_1[1].bias
        v8 = self.module_1
        v9 = v1 * 0.5
        v10 = v1 * v1 * v1
        v11 = v10 * 0.044715
        v12 = v1 + v11
        v13 = v12 * 0.7978845608028654
        v14 = torch.tanh(v13)
        v15 = v14 + 1
        v16 = v9 * v15
        return v16
# Inputs to the model
x1 = torch.randn(3, 4, 4, 4)
