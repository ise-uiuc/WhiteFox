
# TODO: Import pytorch.nn library
# TODO: Implement forward function with pointwise transposed convolution and sigmoid activation function.
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Instantiate a pointwise transposed convolution module with the specified parameters.
        self.conv_t = torch.nn.ConvTranspose2d(157, 78, kernel_size=(1, 16), stride=(1, 66), padding=(0, 11))
    def forward(self, x1):
        # TODO: Implement forward logic
        v1 = self.conv_t(x1)
        v2 = torch.sigmoid(v1)
        return v2

# Inputs to the model
x1 = torch.randn(1, 157, 169, 23)

# Output example of the model
with torch.no_grad():
    test_mod = Model()
    test_mod.eval()
    out = test_mod(x1)
# Output shape should be (1, 78, 4, 1)
print(out.shape)

