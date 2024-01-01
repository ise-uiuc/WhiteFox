
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        # Inception-like 1x1 convolution with bias add
        x1 = F.conv2d(x1, torch.ones(1, 1, 7) * 0.2,
                     torch.zeros(1), stride=1, padding=3,
                     groups=1, bias=torch.ones(1))
        # Average pooling
        x1 = F.avg_pool2d(x1, kernel_size=3, stride=1, padding=1)
        # Inception-like 3x3 convolution with bias add
        x1 = F.conv2d(x1, torch.ones(1, 7, 1) * 0.5,
                     torch.zeros(1), stride=1, padding=1,
                     groups=1, bias=torch.ones(1))
        # Inception-like 5x5 convolution with bias add
        x1 = F.conv2d(x1, torch.ones(1, 7, 1) * 0.5,
                     torch.zeros(1), stride=1, padding=2,
                     groups=1, bias=torch.ones(1))
        # Inception-like 1x1 convolution with bias add
        x1 = F.conv2d(x1, torch.ones(1, 7, 1) * 0.25,
                     torch.zeros(1), stride=1, padding=0,
                     groups=1, bias=torch.ones(1))
        # Inception-like 5x5 convolution with bias add
        x1 = F.conv2d(x1, torch.ones(1, 1, 7) * 0.125,
                     torch.zeros(1), stride=1, padding=2,
                     groups=1, bias=torch.ones(1))
        # Average pooliing
        x1 = F.avg_pool2d(x1, kernel_size=3, stride=1, padding=1)
        # Inception-like 1x1 convolution with bias add
        x1 = F.conv2d(x1, torch.ones(1, 7, 1) * 0.25,
                     torch.zeros(1), stride=1, padding=0,
                     groups=1, bias=torch.ones(1))
        # Inception-like 1x7 convolution with bias add
        x1 = F.conv2d(x1, torch.ones(1, 1, 7) * 0.5,
                     torch.zeros(1), stride=1, padding=0,
                     groups=1, bias=torch.ones(1))
        # Inception-like 7x1 convolution with bias add
        x1 = F.conv2d(x1, torch.ones(1, 7, 1) * 0.5,
                     torch.zeros(1), stride=1, padding=3,
                     groups=1, bias=torch.ones(1))
        # Inception-like 1x7 convolution with bias add
        x1 = F.conv2d(x1, torch.ones(1, 1, 7) * 0.5,
                     torch.zeros(1), stride=1, padding=0,
                     groups=1, bias=torch.ones(1))
        # Max pooling
        x1 = F.max_pool2d(x1, 3, 1, 1)
        # Inception-like 1x1 convolution with bias add.
        x1 = F.conv2d(x1, torch.ones(1, 7, 1) * 0.25,
                     torch.zeros(1), stride=1, padding=0,
                     groups=1, bias=torch.ones(1))

        # Gelu activation
        x1 = F.gelu(x1)
        # Concatenate tensors along one dimension
        x1 = torch.cat([x1, x2])

        # Average pooling
        x1 = F.avg_pool2d(x1, 3, 1, 1)

        # Fully-connected layer with 1440 hidden units and ReLU activation.
        x1 = x1.reshape(x1.size(0), -1)
        x1 = torch.nn.functional.dropout(x1, p=0.8, training=True)
        return x1
# Inputs to the model
x1 = torch.randn(1, 2048, 7, 7)
x2 = torch.randn(1, 2048)
