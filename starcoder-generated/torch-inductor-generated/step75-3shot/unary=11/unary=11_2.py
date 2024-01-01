
model = torch.nn.Sequential(
    OrderedDict([
        ('conv_transpose', torch.nn.ConvTranspose2d(32, 64, 1, stride=1, padding=2, output_padding=1)),
        ('max_pool', torch.nn.MaxPool2d(kernel_size=5, stride=2, padding=3)),
        ('max_unpool', torch.nn.MaxUnpool2d(kernel_size=5, stride=2, padding=3)),
        ('adaptive_avg_pool', torch.nn.AdaptiveAvgPool2d(output_size=17)),
        ('flatten', torch.nn.Flatten()),
        ('linear', torch.nn.Linear(5376, 10))
    ]))
# Inputs to the model
x1 = torch.randn(1,32,28,28)
# Model Ends
