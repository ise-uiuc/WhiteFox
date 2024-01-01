
torch._convolution()

# Inputs to the model
x = torch.randn(64, 3, 224, 112, 112)
split_sizes_0 = [20, 40]
split_sizes_1 = [4, 8, 8]
stride = []
output, _, _, _, _, _, _ = torch._convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled)
