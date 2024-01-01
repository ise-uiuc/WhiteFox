 definition
class Model(nn.Module):
    def __init__(self, input_model, output_model, input_tensor):
        super().__init__()
      ...

# Initializing the model
model_orig = Model_Orig()
model_copy = Model_Copy()

# Inputs to the model
input_tensor = torch.randn(40, 10, 32, 32)
