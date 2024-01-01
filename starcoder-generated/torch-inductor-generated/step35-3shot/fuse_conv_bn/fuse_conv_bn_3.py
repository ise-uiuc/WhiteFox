
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc2 = torch.nn.Linear(32, 2)
        self.fc1 = torch.nn.Linear(2, 2)
    def forward(self, x2, x):
        x = self.fc2(x2)
        x = self.fc1(x)
        return x

    torch.manual_seed(1)
    torch.onnx.export(Model().eval(), (torch.randn(2, 5), torch.randn(2, 2)), "model.onnx", verbose=False, opset_version=9)

    model = Model()
    model.eval()
    torch.manual_seed(1)
    x1 = torch.randn(2, 5)
    torch.manual_seed(1)
    x2 = torch.randn(2, 2)
    torch.manual_seed(1)
    _ = list(model(x1, x2))
    torch.onnx.export(model, (x1, x2), "model.onnx", verbose=False,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['x1', 'x2'], output_names=['output'],
                    dynamic_axes={'x1': {1: 'N'},
                                  'x2': {1: 'N'},
                                  'output': {1: 'N'},
                                  })

    