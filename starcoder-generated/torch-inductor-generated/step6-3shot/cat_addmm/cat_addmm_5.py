
model_file = Path('concat_addmm.onnx')

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.addmm = torch.nn.Linear(6, 6).float()
 
    def forward(self, x1):
        v1 = self.addmm(x1)
        v2 = torch.cat(v1)
        return v2

# Initializing the model
model = Model()

# Inputs to the model
x1 = torch.randn(2, 6)

# Saving the model
torch.onnx.export(model,             # model being run
                  (x1,),            # model input (or a tuple for multiple inputs)
                  model_file.open('wb'), # where to save the model (can be a file or file-like object)
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],   # the model's input names
                  output_names=['output']       # the model's output names
)
