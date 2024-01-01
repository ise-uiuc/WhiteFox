
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key = torch.nn.Parameter(torch.randn(13, 2, 1, 51, 87))
    def forward(self, x1):
        q = x1
        k = self.key
        v = x1
        inv_scale = math.sqrt(k.size(1))
        scaled_dot_product = torch.matmul(q, k.transpose(-2, -1)) / inv_scale
        attention_weights = scaled_dot_product.softmax(dim=-1)
        output = attention_weights.matmul(v)
        return output
# Inputs to the model
x1 = torch.randn(51, 7, 64, 85, 87)
model = Model()
model.eval()
torch.onnx.export(model,               # model being run
            x1,                         # model input (or a tuple for multiple inputs)
            "model_onnx.onnx",   # where to save the model (can be a file or file-like object)
            export_params=True,        # store the trained parameter weights inside the model file
            opset_version=10,          # the ONNX version to export the model to
            do_constant_folding=True,  # whether to execute constant folding for optimization
            input_names = ['input'],   # the model's input names
            output_names = ['output'], # the model's output names
            dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                          'output' : {0 : 'batch_size'}})

