import torch
from PIL import Image
import open_clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from utils.extract_fashioniq_patch import cut_image_4, cut_image_9


device="cpu"


model, _, _ = open_clip.create_model_and_transforms(
    "RN50x4",
    pretrained="openai",
    device=device
)

model.eval()


preprocess = Compose([
    Resize(288),
    CenterCrop(288),
    ToTensor(),
    Normalize(
        (0.48145466,0.4578275,0.40821073),
        (0.26862954,0.26130258,0.27577711)
    )
])


img = Image.open(
    "fashion-iq/images/B005X4PL1G.png"
).convert("RGB")


img = img.resize((360,360))


patches = cut_image_4(img) + cut_image_9(img)


print("Patch number:", len(patches))


features=[]

with torch.no_grad():

    for p in patches:

        x=preprocess(p)
        x=x.unsqueeze(0)

        f=model.encode_image(x)

        features.append(f)


features=torch.cat(features,0)

print(features.shape)


torch.save(
    features.float(),
    "fashion-iq/fashion_local13/B005X4PL1G.pth"
)

print("saved")

