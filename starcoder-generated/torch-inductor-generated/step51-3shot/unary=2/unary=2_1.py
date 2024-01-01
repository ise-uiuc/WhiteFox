
class Model(torch.nn.Module):

    def __init__(self, channels=4, dim=4, input_shape=(8, 8)):
        super(Model, self).__init__()
        self.out_channels = channels
        self.dim = dim
        self.size = input_shape[0]
        for n in range(1, math.floor(math.log(size, 2))):
            self.size *= 2
        layers = []
        layers.append(nn.PReLU(self.dim,0.5))
        convtrs = []
        convtrs.append(nn.ConvTranspose2d(self.dim, self.dim, 2,1,padding=0))
        convtrs.append(nn.Tanh())
        self.layers.append(nn.Sequential(*self.layers))
        main_layer = [self.layers[n] for n in range(len(layers))]
        self.layers = nn.Sequential(*main_layer)
# Inputs to the model
x1 = torch.randn(1, 4, 8, 8)
