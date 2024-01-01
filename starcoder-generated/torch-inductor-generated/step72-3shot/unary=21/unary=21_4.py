
class ModuleTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input1):
        
        module1 = torch.nn.Conv2d(1, 3, kernel_size=(1, 1), stride=(1, 1), dilation=(1, 1))
        module1_out = module1(input1)
        atg_out_0 = torch.sign(module1_out)
        atg_out_1 = torch.sigmoid(atg_out_0)
        atg_out_2 = torch.tanh(atg_out_1)
        atg_out_3 = torch.abs(atg_out_2)
        module3 = torch.nn.Conv2d(3, 1, kernel_size=(1, 1), stride=(1, 1), dilation=(1, 1))
        module3_out = module3(atg_out_3)
        atg_out_4 = torch.tanh(module3_out)
        return atg_out_4
# Inputs to the model
input1 = torch.randn(1, 1, 224, 224)
