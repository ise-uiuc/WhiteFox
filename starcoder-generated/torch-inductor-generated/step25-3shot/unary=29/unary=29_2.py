
class Model(torch.nn.Module):
    def __init__(self,min_value=1.2, max_value= 6.0):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose1d(32,32,kernel_size=3,stride=1,padding=1)
        self.conv_transpose2 = torch.nn.ConvTranspose1d(32,64,kernel_size=3,stride=1,padding=1)
        self.conv_transpose3 = torch.nn.ConvTranspose1d(64,64,kernel_size=5,stride=1,padding=2)
        self.conv_transpose4 = torch.nn.ConvTranspose1d(64,64,kernel_size=3,stride=1,padding=1)
        self.conv_transpose1 = torch.nn.ConvTranspose1d(32,32,kernel_size=3,stride=1,padding=1)
        self.conv_transpose2 = torch.nn.ConvTranspose1d(32,64,kernel_size=3,stride=1,padding=1)
        self.conv_transpose3 = torch.nn.ConvTranspose1d(64,64,kernel_size=5,stride=1,padding=2)
        self.conv_transpose4 = torch.nn.ConvTranspose1d(64,64,kernel_size=3,stride=1,padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x):
        v1 = self.conv_transpose1(x)
        v3 = torch.clamp_min(v1,self.min_value)
        v4 = self.conv_transpose2(v3)
        v6 = torch.clamp_max(v4,self.max_value)
        v7 = self.conv_transpose3(v6)
        v9 = torch.clamp_min(v7,self.min_value)
        v10 = self.conv_transpose3(v9)
        v12 = torch.clamp_max(v10,self.max_value)
        v13 = self.conv_transpose3(v12)
        v15 = torch.clamp_min(v13,self.min_value)
        v16 = self.conv_transpose3(v15)
        v18 = torch.clamp_max(v16,self.max_value)
        return v18
# Input to the model
x1 = torch.randn(1,32,2048)
