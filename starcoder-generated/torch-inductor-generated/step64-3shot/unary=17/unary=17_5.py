
class Model(torch.nn.Module):
    def __init__(self):
       super().__init__()
       self.conv_transpose = torch.nn.ConvTranspose2d(9, 21, (5, 5), stride=(5, 5), padding=(0, 0), output_padding=(1, 1), groups=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.relu(v1)
        v3 = torch.sigmoid(v2)
        return v3    
# Inputs to the model
x1 = torch.randn(1, 9, 50, 50)
