import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import urllib.request
from PIL import Image

# Torch
import torch
import torchvision
import torchvision.transforms as TS

# Grounding DINO
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import (
    clean_state_dict,
    get_phrases_from_posmap,
)

# Segmentate Anything
from segment_anything import build_sam, SamPredictor

# Recognize Anything
from recognize_anything.models import ram


class SegmentateImage(object):
    def __init__(self):
        # TODO: Move to config folder instead of hardcoding them here.
        self.box_threshold = 0.25
        self.text_threshold = 0.2
        self.iou_threshold = 0.5
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_size = 384
        groundingdino_config_file = "config/groundingdino_cfg.py"
        self.output_dir = "outputs"
        os.makedirs(self.output_dir, exist_ok=True)

        # Checkpoints
        checkpoints = self.download_checkpoints()
        # Load grounding model
        self.groundino_model = self.load_groundino_model(
            config_path=groundingdino_config_file,
            checkpoint_path=checkpoints.grounded,
        )
        # Load tag2text model
        self.ram_model = self.load_ram_model(checkpoint_path=checkpoints.ram)
        # Load SAM predictor
        self.sam_predictor = SamPredictor(
            build_sam(checkpoint=checkpoints.sam).to(self.device)
        )

    def download_checkpoints(self):
        checkpoints = SLConfig.fromfile("./config/checkpoints.py")
        checkpoints = [
            (checkpoints.ram_url, checkpoints.ram),
            (checkpoints.grounded_url, checkpoints.grounded),
            (checkpoints.sam_url, checkpoints.sam),
        ]
        # Download all models and save them to the models folder.
        for i, (url, file_name) in enumerate(checkpoints):
            file_name = os.path.join("models", file_name)
            checkpoints[i] = (url, file_name)
            urllib.request.urlretrieve(url, file_name)

        return checkpoints

    def get_grounding_output(self, model, image, tags):
        tags = tags.lower()
        tags = tags.strip()
        if not tags.endswith("."):
            tags = tags + "."
        model = model.to(self.device)
        image = image.to(self.device)
        with torch.no_grad():
            outputs = model(image[None], captions=[tags])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        logits.shape[0]

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > self.box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        logits_filt.shape[0]

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(tags)
        # build pred
        pred_phrases = []
        scores = []
        for logit, _ in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(
                logit > self.text_threshold, tokenized, tokenlizer
            )
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            scores.append(logit.max().item())

        return boxes_filt, torch.Tensor(scores), pred_phrases

    def load_ram_model(self, checkpoint):
        model = ram(
            pretrained=checkpoint,
            image_size=self.image_size,
            vit="swin_l",
        )
        model.eval()
        model = model.to(self.device)
        return model

    def load_groundino_model(self, config_path, checkpoint):
        args = SLConfig.fromfile(config_path)
        args.device = self.device
        model = build_model(args)
        checkpoint = torch.load(checkpoint, map_location="cpu")
        load_res = model.load_state_dict(
            clean_state_dict(checkpoint["model"]), strict=False
        )
        print(load_res)
        _ = model.eval()
        return model

    def get_objects(self, image_path):
        # Load image
        image_pil, image = self.load_image(image_path)
        # Visualize raw image
        image_pil.save(os.path.join(self.output_dir, image_path + "_raw.jpg"))

        normalize = TS.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = TS.Compose([TS.Resize((384, 384)), TS.ToTensor(), normalize])
        raw_image = image_pil.resize((384, 384))
        raw_image = transform(raw_image).unsqueeze(0).to(self.device)

        # Get tags using ram model
        english_tags, chinese_tags = self.get_tags(image)
        tags = english_tags.replace(" |", ",")
        print("Image Tags: ", tags)

        # Get boxes using grounding model
        boxes_filt, pred_phrases = self.get_boxes(
            image=image,
            image_pil=image_pil,
            tags=tags,
        )

        # Get masks using sam model
        masks = self.segmentate_image(image_path, boxes_filt)

        # Draw output image
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        for mask in masks:
            self.show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        boxes = []
        labels = []
        for box, label in zip(boxes_filt, pred_phrases):
            boxes.append(box)
            labels.append(label)
            self.show_box(box.numpy(), plt.gca(), label)

        plt.title(f"Image Tags: {tags}")
        plt.axis("off")
        plt.savefig(
            os.path.join(self.output_dir, image_path + "_labeled.jpg"),
            bbox_inches="tight",
            dpi=300,
            pad_inches=0.0,
        )

        self.save_mask_data(self.output_dir, tags, masks, boxes_filt, pred_phrases)
        # TODO: Return the masks (points) with the labels.
        return masks, boxes, labels

    def get_boxes(self, image, image_pil, tags):
        boxes_filt, scores, pred_phrases = self.get_grounding_output(
            model=self.groundino_model,
            image=image,
            tags=tags,
        )

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        # use NMS to handle overlapped boxes
        print(f"Before NMS: {boxes_filt.shape[0]} boxes")
        nms_idx = (
            torchvision.ops.nms(boxes_filt, scores, self.iou_threshold).numpy().tolist()
        )
        boxes_filt = boxes_filt[nms_idx]
        pred_phrases = [pred_phrases[idx] for idx in nms_idx]
        print(f"After NMS: {boxes_filt.shape[0]} boxes")
        return boxes_filt, pred_phrases

    def get_tags(self, image):
        tags, tags_chinese = self.ram_model.generate_tag(image)

        return tags[0], tags_chinese[0]

    def segmentate_image(self, image_path, boxes_filt):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.sam_predictor.set_image(image)

        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(
            boxes_filt, image.shape[:2]
        ).to(self.device)

        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(self.device),
            multimask_output=False,
        )
        return masks

    def load_image(self, image_path):
        image_pil = (
            Image.open(image_path)
            .convert("RGB")
            .resize((self.ram_args.image_size, self.ram_args.image_size))
        )
        image = self.ram_transform(image_pil).unsqueeze(0).to(self.device)

        normalize = TS.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = TS.Compose(
            [
                TS.Resize((self.ram_args.image_size, self.ram_args.image_size)),
                TS.ToTensor(),
                normalize,
            ]
        )
        image = transform(image_pil).unsqueeze(0).to(self.device)
        return image_pil, image

    def save_mask_data(self, output_dir, tags, mask_list, box_list, label_list):
        value = 0  # 0 for background

        mask_img = torch.zeros(mask_list.shape[-2:])
        for idx, mask in enumerate(mask_list):
            mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
        plt.figure(figsize=(10, 10))
        plt.imshow(mask_img.numpy())
        plt.axis("off")
        plt.savefig(
            os.path.join(output_dir, "mask.jpg"),
            bbox_inches="tight",
            dpi=300,
            pad_inches=0.0,
        )

        json_data = {
            "tags": tags,
            "mask": [{"value": value, "label": "background"}],
        }
        for label, box in zip(label_list, box_list):
            value += 1
            name, logit = label.split("(")
            logit = logit[:-1]  # the last is ')'
            json_data["mask"].append(
                {
                    "value": value,
                    "label": name,
                    "logit": float(logit),
                    "box": box.numpy().tolist(),
                }
            )
        with open(os.path.join(output_dir, "label.json"), "w") as f:
            json.dump(json_data, f)

    def show_box(self, box, ax, label):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(
            plt.Rectangle(
                (x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2
            )
        )
        ax.text(x0, y0, label)

    def show_mask(self, mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)


def main():
    segmentate_image = SegmentateImage()
    # Test
    segmentate_image.segmentate_image(image_path="config/moveit_test_2.png")


if __name__ == "__main__":
    main()
