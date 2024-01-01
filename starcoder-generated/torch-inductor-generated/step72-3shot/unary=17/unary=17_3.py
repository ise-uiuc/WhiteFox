
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        def relu_activation(args):
            input, output = args
            return torch.nn.ReLU(inplace=False)(input)

        def sigmoid_activation(args):
            input, output = args
            return torch.nn.Sigmoid(inplace=True)(input)
        
        def block(in_filters, out_filters, kernel_size=(3, 3), stride=(1, 1), padding=0):
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_filters, out_filters, kernel_size = kernel_size, stride = stride, padding = padding),
                torch.nn.ReLU(inplace=False)
            )
        self.layers = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3, 32, 3, stride = 2),
            block(32, 64),
            block(64, 128), 
            block(128, 256),
            sigmoid_activation((256, 1))
        )
    def forward(self, x1):
        return self.layers(x1)
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
