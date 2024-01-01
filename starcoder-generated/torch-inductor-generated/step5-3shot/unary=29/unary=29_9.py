
        conv_transpose = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=1)
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
