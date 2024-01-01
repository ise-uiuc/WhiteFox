
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d()
    def forward(self):
        x1 = torch.ones([1,1,224,224])
        return self.conv_t(x1)
# Inputs to the model
