import torch 
from vit_pytorch import ViT
from vit_pytorch.simmim import SimMIM
from vit_pytorch.recorder import Recorder
for i in range(5):
    v = ViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 8,
        mlp_dim = 2048
    )

    mim = SimMIM(
        encoder = v,
        masking_ratio = 0.5 
    )
    v = Recorder(v)
    images = torch.randn(8, 3, 256, 256)
    loss = mim(images)
    loss.backward()
    preds, attns = v(images)
    print(preds)
    print(attns)

torch.save(v.state_dict(), './trained-vit.pt')
print(v.state_dict())


