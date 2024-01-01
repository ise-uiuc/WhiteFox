
import transformers
class Model(transformers.PreTrainedModel):
    def __init__(self):
        super().__init__(transformers.RobertaConfig())
        # Define an operator that does a tensor product
        self.operator_1 = torch.nn.Linear(in_features=5, out_features=3)
        # Define an operator with a different input tensor
        self.operator_2 = torch.nn.Linear(in_features=6, out_features=3)
    def forward(self, x):
        o1 = self.operator_1(x)
        o2 = self.operator_2(x)
        o =torch.cat([o1, o2], dim=1)
        y = o.view(o.shape[0], -1)
        return y
# Inputs to the model
x = torch.randn(2, 3, 5)
