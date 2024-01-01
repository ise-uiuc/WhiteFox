
    def forward(self, x3):
        y = torch.matmul(x3, x3)
        z = torch.matmul(x3, x3)
        return y + z + y + y + z, z
# Inputs to the model
x3 = torch.randn(1, 2, 20, 20, 10)
