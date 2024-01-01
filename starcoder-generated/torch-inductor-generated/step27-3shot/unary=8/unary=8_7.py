
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1_transpose = torch.nn.ConvTranspose2d(in_channels=17, 
                                                        out_channels=4, 
                                                        kernel_size=(3, 5),
                                                        stride=(3, 4), 
                                                        padding=(1, 2),
                                                        dilation=2,
                                                        output_padding=3)
        
        self.conv2_transpose = torch.nn.ConvTranspose2d(in_channels=4, 
                                                        out_channels=9, 
                                                        kernel_size=(5, 3),
                                                        stride=(2, 2), 
                                                        padding=(2, 1),
                                                        dilation=2,
                                                        output_padding=1)
    
    def forward(self, x1):
        v1 = self.conv1_transpose(x1)
        v2 = v1 + 3        
        v3 = torch.clamp(v2, min=0)
        v4 = torch.clamp(v3, max=6)
        v5 = v1 * v4
        v6 = v5 / 6
        
        v7 = self.conv2_transpose(v6)
        v8 = v7 + 3        
        v9 = torch.clamp(v8, min=0)
        v10 = torch.clamp(v9, max=6)
        v11 = v7 * v10
        v12 = v11 / 6
        
        return v12
# Inputs to the model
x1 = torch.randn(1, 17, 32, 32)
