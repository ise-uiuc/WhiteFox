
model = Model()
q = torch_glow.get_execution_engine(model)
f = io.BytesIO()
torch.onnx.export(model, x3, f, verbose=False, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
model_bytes = f.getvalue()
# Inputs to the model
input_dict = dict()
x3_traced = torch.randn(19, 12, 2, 63)
for name, value in x3_traced.items():
    input_dict[name] = value
