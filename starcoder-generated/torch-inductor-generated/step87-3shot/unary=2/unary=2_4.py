
# class Module(torch.nn.Module):
#     def __init__(self):
#         super(Module,self).__init__()    
#         self.conv1 = torch.nn.Conv2d(10, 5, kernel_size = 5, stride = 1, padding = 2, dilation = 1, groups=1, bias=True)    
#         self.convt1 = torch.nn.ConvTranspose2d(5, 10, kernel_size = 5, stride = 1, padding = 2, dilation=1, groups = 1, output_padding = 0, bias=True)    
#     def forward(self, x1):
#         v1 = self.conv1(x1)     
#         f1 = torch.randperm(v1.shape[0], device = v1.device) 
#         t1 = torch.zeros(v1.shape) if f1[0] - f1[1] else torch.ones(v1.shape) 
#         v2 = v1 * t1    
#         v3 = torch.abs(v2)  
#         v4 = torch.tanh(v3)     
#         v5 = torch.tanh(v4)     
#         v6 = v5 * t1     
#         v7 = self.convt1(v6)     
#         return v7
# Inputs to the model
x1 = torch.randn(10, 10, 32, 32)
