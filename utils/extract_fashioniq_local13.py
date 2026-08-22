import os
import torch
from PIL import Image
from tqdm import tqdm
import glob
import open_clip

from utils.extract_fashioniq_patch import (
    cut_image_4,
    cut_image_9,
    targetpad_transform
)


device="cpu"   # đổi cuda nếu có GPU


model, _, preprocess_clip = open_clip.create_model_and_transforms(
    "RN50x4",
    pretrained="openai",
    device=device
)

model.eval()


root="./fashion-iq/images"

save_root="./fashion-iq/fashion_local13"

os.makedirs(save_root, exist_ok=True)


image_paths = glob.glob(root+"/*.png")


print("Images:",len(image_paths))


preprocess = targetpad_transform(
    target_ratio=1.25,
    dim=288
)


with torch.no_grad():

    for path in tqdm(image_paths):

        name=os.path.basename(path).replace(".png","")

        save_path=os.path.join(
            save_root,
            name+".pth"
        )

        if os.path.exists(save_path):
            continue


        image=Image.open(path).convert("RGB")

        image=image.resize((360,360))


        patches=cut_image_4(image)+cut_image_9(image)


        feats=[]

        for patch in patches:

            x=preprocess(patch)
            x=x.unsqueeze(0).to(device)

            feat=model.encode_image(x)

            feats.append(feat)


        feats=torch.cat(feats,dim=0)

        # [13,640]
        feats=feats.float().cpu()


        torch.save(
            feats,
            save_path
        )
