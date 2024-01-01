
class Model(torch.nn.Module):
    # The following class variables must be defined for all models:
    #     opset_version: int = 9 or 11
    #     ir_version: int = 3 or 4
    #     producer_name: str = "onnx-example-producer"
    #     producer_version: str = "0.0.1"
    #     domain: str = "custom-domain" or "ai.onnx"

    def __init__(self, negative_slope):
        super().__init__()
        self.linear = torch.nn.Linear(128, 256)
        self.negative_slope = negative_slope
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
    
# Initializing the model with negative_slope = 0.01
m = Model(0.01)

# Inputs to the model
x1 = torch.randn(1, 128)
