
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()   
        # Use hardtanh to clamp input values in the range min_val to max_val (default: min_val=0, max_val=1)
        self.conv1 = nn.ConvTranspose2d(1, 1, 5, padding=2)
        self.conv2 = nn.ConvTranspose2d(1, 1, 5, padding=2)
        self.conv3 = nn.ConvTranspose2d(1, 1, 5, padding=2)
        self.conv4 = nn.ConvTranspose2d(1, 1, 5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x1 = torch.sigmoid(self.pool1(F.hardtanh(self.conv1(x))))
        x2 = torch.sigmoid(self.pool1(F.hardtanh(self.conv2(x1))))
        x3 = torch.sigmoid(self.pool1(F.hardtanh(self.conv3(x2))))
        x4 = torch.sigmoid(self.pool1(F.hardtanh(self.conv4(x3))))
        x5 = x4
        return x5

input_size = (1, 1, 4, 4)
batch_size = 33

model = Model().eval()
x = torch.randn(batch_size, *input_size)
scripted_model = torch.jit.script(model)

def generate_model(model):
    scripted_model = torch.jit.script(model)
    model = copy.deepcopy(model)
    torch.onnx.export(model, x, "model_transposed_conv_%f.onnx" % 0.214, verbose=False, use_dynamic_axes=False,
                      opset_version=11, do_constant_folding=False, input_names=['x'])
    torch.onnx.export(scripted_model, x, "scripted_transposed_conv_%f.onnx" % 0.214, verbose=False,
                      use_dynamic_axes=False, opset_version=11, do_constant_folding=False, input_names=['x'])
    os.system('mo --input_model=model_transposed_conv_%f.onnx --output_dir=model_transposed_conv_%f_FP32'
              '--data_type=FP32 --input=x --output=0 --scale_values=x:255' % (0.214, 0.214))
    os.system('mo --input_model=scripted_transposed_conv_%f.onnx --output_dir=scripted_transposed_conv_%f_FP32'
              '--data_type=FP32 --input=x --output=0 --scale_values=x:255' % (0.214, 0.214))
    os.system('mo --input_model=model_transposed_conv_%f.onnx --output_dir=model_transposed_conv_%f_FP16'
              '--data_type=FP16 --input=x --output=0 --scale_values=x:255' % (0.214, 0.214))
    os.system('mo --input_model=scripted_transposed_conv_%f.onnx --output_dir=scripted_transposed_conv_%f_FP16'
              '--data_type=FP16 --input=x --output=0 --scale_values=x:255' % (0.214, 0.214))
    return

generate_model(model)
