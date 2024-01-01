
class Model(torch.nn.Module):
    def forward(self, x1):
        z1 = torch.add(torch.nn.functional.dropout(x1, 0., False, False), torch.tensor([40], dtype=torch.float64))
        z2 = torch.add(z1, 1.5)
        s1 = torch.pow(z2, 2)
        x2 = torch.add(torch.nn.functional.interpolate(s1, [30], mode='linear', align_corners=True), 0.6)
        x3 = torch.add(torch.nn.functional.silu(x2), 21)
        x4 = torch.nn.functional.adaptive_avg_pool2d(x3, (769,))
        x5 = torch.nn.functional.silu(x4, self.a61, self.a61214)


model = Model()
model.load_state_dict(
            slim.get_state_dict("resnet50", torch_hub_url("google/resnet_v1", pretrained=True, num_classes=100))['model_state_dict']
        )
# Inputs to the model
x1 = torch.randn((1, 3, 4))
