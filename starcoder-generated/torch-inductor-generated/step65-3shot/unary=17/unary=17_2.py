
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.module1 = torch.nn.Sequential(
            (
                "conv_transpose",
                torch.nn.ConvTranspose2d(1, 1, kernel_size=(3, 3), padding=(2, 2)),
            ),
            (
                "relu",
                torch.nn.ReLU(),
            )
        )
        self.module2 = torch.nn.Sequential(
            (
                "conv_transpose1",
                torch.nn.ConvTranspose2d(1, 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            ),
            (
                "relu1_2",
                torch.nn.ReLU(),
            )
        )
    def forward(self, x1):
        v1 = self.module1.conv_transpose(x1)
        v2 = self.module1.relu(v1)
        v3 = self.module2.conv_transpose1(v2)
        v4 = self.module2.relu1_2(v3)
        v5 = torch.add(v4, 1)
        for module in self.modules():
            for param in module.parameters(recurse=False):
                # To avoid torch.nn.Sequential parameters are not registered as parameters in Model class, so we need to register them here
                param

        v6 = torch.sigmoid(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 1, 3, 3)
