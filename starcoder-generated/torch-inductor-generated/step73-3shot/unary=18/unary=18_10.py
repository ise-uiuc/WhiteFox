
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ConvA1 = torch.nn.Conv2d(in_channels=3, out_channels=48, kernel_size=(1,1), padding=(0,0), bias=False)
        self.ConvA2 = torch.nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(3,3), padding=(1,1), bias=False)
        self.ConvA3 = torch.nn.Conv2d(in_channels=64, out_channels=48, kernel_size=(1,1), padding=(0,0), bias=False)
        
        self.ConvB1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(1,1), padding=(0,0), bias=False)
        self.ConvB2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), padding=(1,1), bias=False)
        self.ConvB3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5,5), padding=(2,2), bias=False)
        
        self.Avg_pooling = torch.nn.AvgPool2d(kernel_size=(3,3), stride=(2,2), padding=(0,0), ceil_mode=False, count_include_pad=True, divisor_override=None)
        self.ConvC1 = torch.nn.Conv2d(in_channels=112, out_channels=64, kernel_size=(1,1), padding=(0,0), bias=False)
        self.ConvC2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=(1,1), bias=False)
        self.ConvC3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1,1), padding=(0,0), bias=False)
        self.ConvC4 = torch.nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(1,1), padding=(0,0), bias=False)
        
        self.ConvD1 = torch.nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(1,1), padding=(0,0), bias=False)
        self.ConvD2 = torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3,3), padding=(1,1), bias=False)
        self.ConvD3 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1,1), padding=(0,0), bias=False)
    def forward(self, inp):
        o1 = self.ConvA3(self.ConvA2(self.ConvA1(inp)))
        o2 = self.ConvB3(self.ConvB2(self.ConvB1(inp)))    
        o3 = self.Avg_pooling(torch.cat((o1, o2), 1))
        
        o4 = self.ConvC4(self.ConvC3(self.ConvC2(self.ConvC1(o3))))
        o5 = self.ConvD3(self.ConvD2(self.ConvD1(o3)))
        
        return o1, o2, o3, o4, o5
# Inputs to the model
x3 = torch.randn(1, 3, 224, 224)
