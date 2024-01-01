
class Model(torch.nn.Module):
    def __init__(self, w_in):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(w_in, 3, (3, 7), stride=2, padding=4)
    def forward(self, x):
        x1 = self.conv_t(x)
        return x1
w_in = random.randint(20, 60)
input_shape = (64, w_in, 32, 32)
# Inputs to the model
x = torch.randn(*input_shape)
