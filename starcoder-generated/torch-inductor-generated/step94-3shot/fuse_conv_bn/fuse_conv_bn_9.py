
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op0 = torch.nn.Conv2d(3, 3, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.op1 = torch.ops.torch_ipex.conv_transpose2d(7, 7, kernel_size=(16,16), stride=(3,3), padding=(2,2), output_padding=(1,1), output_size=(18,18))
        self.op2 = torch.ops.torch_ipex.dropout(0.1)
        self.op3 = torch.nn.BatchNorm2d(640)
        self.op4 = torch.nn.ReLU()
        self.op5 = torch.nn.ConvTranspose2d(64, 64, kernel_size=(2,2), stride=(2,2), padding=(0,0))
        self.op6 = torch.nn.ReLU()
        self.op7 = torch.nn.ConvTranspose2d(64, 3, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.op8 = torch.ops.torch_ipex.flatten(1, -1)
        self.op9 = torch.nn.Linear(324, 10)
        self.op10 = torch.nn.LogSoftmax(dim=-1)
        self.flatten = torch.nn.Flatten()
    def forward(self, x1):
        opt_conv_fused, opt_bn_fused, x = torch._C._jit_pass_fuse_conv_bn(self.op0, self.op1)
        opt_conv_fused = torch._C._jit_pass_conv_pack(opt_conv_fused, x)
        x = self.op0(x)
        opt_conv_fused, opt_bn_fused, x = torch._C._jit_pass_fuse_conv_bn(self.op3, self.op4)
        opt_conv_fused = torch._C._jit_pass_conv_pack(opt_conv_fused, x)
        x = self.op1(x)
        x = self.op3(x)
        x = self.op4(x)
        return self.op10(self.op9(x))
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
