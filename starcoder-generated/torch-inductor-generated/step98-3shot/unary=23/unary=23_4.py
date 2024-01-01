
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 3, 1)
    def forward(self, x1):
        v2 = torch.tanh(x1.flatten(start_dim=1) # Unsqueeze x1 so that its shape would be (N, 1, L, C)
        v1 = self.conv_transpose(v2)
        v3 = v1.view(x1.shape)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 8, 8)
