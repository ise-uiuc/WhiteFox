
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(3, 2, stride=1, groups=3)
    def forward(self, x):
        # This operation is not fused because the groups attribute is not supported by the version of F::batch_norm that torch::onnx::export invokes.
        x1 = F.batch_norm(self.conv2d(x), running_mean=torch.zeros(2), running_var=torch.ones(2), weight=torch.ones(2), bias=torch.zeros(2), training=torch.onnx.is_in_onnx_export())
        return x1
# Inputs to the model
x1 = torch.randn(1, 3, 2, 2)
