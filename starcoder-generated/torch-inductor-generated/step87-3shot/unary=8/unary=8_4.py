
        self.conv_transpose1 = torch.nn.ConvTranspose2d(37, 5, 5, stride=2, padding=1, output_padding=0)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(5, 12, 3, stride=4, padding=0, output_padding=0)
# Inputs to the model
x1 = torch.randn(1, 37, 75, 75)
