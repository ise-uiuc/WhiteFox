
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gelu = torch.nn.GELU()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(3, 4, 3, stride=1, padding=1, bias=False)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(4, 4, 3, stride=1, padding=1, bias=False)
        self.conv_transpose3 = torch.nn.ConvTranspose2d(4, 8, 3, stride=1, padding=1, bias=False)
    def forward(self, x):
        v1 = self.conv_transpose1(x)
        v2 = self.gelu(v1)
        v3 = self.conv_transpose2(v2)
# Inputs to the model
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gelu = torch.nn.GELU()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(3, 4, 3, stride=1, padding=1, bias=False)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(4, 4, 3, stride=1, padding=1, bias=False)
        self.conv_transpose3 = torch.nn.ConvTranspose2d(4, 8, 3, stride=1, padding=1, bias=False)
    def forward(self, x):
        v1 = self.conv_transpose1(x)
        v2 = self.gelu(v1)
        v3 = self.conv_transpose3(v2)
# Inputs to the model
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gelu = torch.nn.GELU()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(3, 4, 3, stride=1, padding=1, bias=False)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(4, 8, 3, stride=1, padding=1, bias=False)
        self.conv_transpose3 = torch.nn.ConvTranspose2d(8, 8, 3, stride=1, padding=1, bias=False)
    def forward(self, x):
        v1 = self.conv_transpose1(x)
        v2 = self.gelu(v1)
        v4 = self.conv_transpose3(v2)
# Inputs to the model
