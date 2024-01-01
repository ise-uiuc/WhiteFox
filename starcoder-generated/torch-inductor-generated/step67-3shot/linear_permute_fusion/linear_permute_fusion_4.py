
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense_4_input_conv2d_transpose_weight = torch.nn.Parameter(torch.randn(48, 128, 3, 3, device='cuda'))
        self.conv2d_5_weight = torch.nn.Parameter(torch.randn(3, 3, 3, 3, device='cuda'))
    def forward(self, input, ):
        conv_4_out = torch.transpose(input, 2, 3)
        dense_4_bias = torch.nn.functional.linear(conv_4_out, self.dense_4_input_conv2d_transpose_weight, torch.nn.functional.linear(input, self.dense_4_input_conv2d_transpose_weight))
        conv2d_5_bias = torch.nn.functional.conv2d(conv_4_out, self.conv2d_5_weight, bias=dense_4_bias)
        return conv2d_5_bias
# Inputs to the model
input = torch.randn(1, 128, 32, 32, device='cuda')
