
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(32, 64, kernel_size=[12], bias=False, groups=4, stride=[1, 2], padding=[0, 5], dilation=[1, 1], padding_mode='zeros')
        torch.manual_seed(1)
        self.bn = torch.nn.BatchNorm2d(64, affine=False, track_running_stats=True)
    def forward(self, x2):
        x3 = self.conv(x2)
        x4 = self.bn(x3)
        return x4
input_name = ["x2"]
output_name = ["x4"]
x2 = torch.rand(1, 32, 256, 320)
model_path = "/tmp/test_model.pth"

# Save the model
torch.jit.save(torch.jit.script(m), model_path)
model_script = torch.jit.load(model_path)
save_input(x2, input_name)
save_output(output_name)

# Run the model and get output
print(model_script(x2))
run_model(model_script, input_name, output_name)
# Output: tensor([[-1.4991, -0.6116,  0.1356]], grad_fn=<SliceBackward0>)
