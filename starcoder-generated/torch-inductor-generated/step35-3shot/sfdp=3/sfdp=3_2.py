
class Model(torch.nn.Module):
    def __init__(self, input1: torch.Tensor, input2: torch.Tensor):
        super().__init__()
        self.mul_1 = torch.quantization.fuse_modules([input1, input2], [['bn','relu'],'mul'], inplace=False)
        self.mul_2 = torch.quantization.fuse_modules([input1, self.mul_1], [['conv_1', 'bn_1'],'mul'], inplace=False)
        self.mul_3 = torch.quantization.fuse_modules([self.mul_1, self.mul_2], [['conv_2', 'bn_2'],'mul'], inplace=False)
        self.softmax_qk = torch.nn.softmax(torch.matmul(self.mul_3, torch.transpose(self.mul_3, -2, -1)), dim=-1)
        self.dropout_qk = torch.quantization.fuse_modules([self.mul_3, self.softmax_qk], ['softmax','mul'], inplace=False)
        self.dropout_qk = torch.nn.dropout(self.dropout_qk, p=0.3)
 
    def forward(self, x1):
        return self.dropout_qk.matmul(x1)

class Model(torch.nn.Module):
    def __init__(self, input1: torch.Tensor, input2: torch.Tensor):
        super().__init__()
        self.mul_1 = torch.quantization.fuse_modules([input1, input2], [['bn','relu'],'mul'], inplace=False)
        self.mul_2 = torch.quantization.fuse_modules([input1, self.mul_1], [['conv_1', 'bn_1'],'mul'], inplace=False)
        self.mul_3 = torch.quantization.fuse_modules([self.mul_1, self.mul_2], [['conv_2', 'bn_2'],'mul'], inplace=False)
        self.softmax_qk = torch.nn.softmax((torch.matmul(self.mul_3, torch.transpose(self.mul_3, -2, -1)) * 3.0517578125e-05), dim=-1)
        self.dropout_qk = torch.quantization.fuse_modules([self.mul_3, self.softmax_qk], ['softmax','mul'], inplace=False)
        self.dropout_qk = torch.nn.dropout(self.dropout_qk, p=0.3)
 
    def forward(self, x1):
        return self.dropout_qk.matmul(x1)

class Model(torch.nn.Module):
    def __init__(self, input1: torch.Tensor, input2: torch.Tensor):
        super().__init__()
        self.mul_1 = torch.quantization.fuse_modules([input1, input2], [['bn','relu'],'mul'], inplace=False)
        self.mul_2 = torch.quantization.fuse_modules([input1, self.mul_1], [['conv_1', 'bn_1'],'mul'], inplace=False)
        self.mul_3 = torch.quantization.fuse_modules([self.mul_1, self.mul_2], [['conv_2', 'bn_2'],'mul'], inplace=False)
        self.softmax_qk = torch.nn.softmax((torch.matmul(self.mul_3, torch.transpose(self.mul_3, -2, -1)) * 1.74537890625e-05), dim=-1)
        self.dropout_qk = torch.quantization.fuse_modules([self.mul_3, self.softmax_qk], ['softmax','mul'], inplace=False)
        self.dropout_qk = torch.nn.dropout(self.dropout_qk, p=0.3)
 
    def forward(self, x1):
        return self.dropout_qk.matmul(x1)

class Model(torch.nn.Module):
    def __init__(self, input1: torch.Tensor, input2: torch.Tensor):
        super().__init__()
        self.mul_1 = torch.quantization.fuse_modules([input1, input2], [['bn','relu'],'mul'], inplace=False)
        self.mul_2 = torch.quantization.fuse_modules([input1, self.mul_1], [['conv_1', 'bn_1'],'mul'], inplace=False)
        self.mul_3 = torch.quantization.fuse_modules([self.mul_1, self.mul_2], [['conv_2', 'bn_2'],'mul'], inplace=False)
        self.softmax_qk = torch.nn.softmax((torch.matmul(self.mul_3, torch.transpose(self.mul_3, -2, -1)) * 3.72529029846e-08), dim=-1)
        self.dropout_qk = torch.quantization.fuse_modules([self.mul_3, self.softmax_qk], ['softmax','mul'], inplace=False)
        self.dropout_qk = torch.nn.dropout(self.dropout_qk, p=0.3)
 
    def forward(self, x1):
        return self.dropout_qk.matmul(x1)

class Model(torch.nn.Module):
    def __init__(self, input1: torch.Tensor, input2: torch.Tensor):
        super().__init__()
        self.mul_1 = torch.quantization.fuse_modules([input1, input2], [['bn','relu'],'mul'], inplace=False)
        self.mul_2 = torch.quantization.fuse_modules([input1, self.mul_1], [['conv_1', 'bn_1'],'mul'], inplace=False)
        self.mul_3 = torch.quantization.fuse_modules([self.mul_1, self.mul_2], [['conv_2', 'bn_2'],'mul'], inplace=False)
        self.softmax_qk = torch.nn.softmax((torch.matmul(self.mul_3, torch.transpose(self.mul_3, -2, -1)) * 5.27587890625e-06), dim=-1)
        self.dropout_qk = torch.quantization.fuse_modules([self.mul_3, self.softmax_qk], ['softmax','mul'], inplace=False)
        self.dropout_qk = torch.nn.dropout(self.dropout_qk, p=0.3)
 
    def forward(self, x1):
        return self.dropout_qk.matmul(x1)

class Model(torch.nn.Module):
    def __init__(self, input1: torch.Tensor, input2: torch.Tensor):
        super().__init__()
        self.mul_1 = torch.quantization.fuse_modules([input1, input2], [['bn','relu'],'mul'], inplace=False)
        self.mul_2 = torch.quantization.fuse_modules([input1, self.mul_1], [['conv_1', 'bn_1'],'mul'], inplace=False)
        self.mul_3 = torch.quantization.fuse_modules([self.mul_1, self.mul_2], [['conv_2', 'bn_2'],'mul'], inplace=False)
        self.softmax_qk = torch.nn.softmax((torch.matmul(self.mul_3, torch.transpose(self.mul_3, -2, -1)) * 2.0343017578125e-05), dim=-1)
        self.dropout_qk = torch.quantization.fuse_modules([self.mul_3, self.softmax_qk], ['softmax','mul'], inplace=False)
        self.dropout_qk = torch.nn.dropout(self.dropout_qk, p=0.3)
 
    def forward(self, x1):
        return self.dropout_qk.matmul(x1)

# Initializing the model
m = Model(torch.randn(2, 2), torch.randn(2, 2))

# Inputs to the model
x1 = torch.randn(1, 2, 4, 4)
