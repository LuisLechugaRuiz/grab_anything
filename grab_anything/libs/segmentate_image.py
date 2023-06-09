import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import requests
from tqdm import tqdm
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

# Recognize Anything - TODO: Install by pip.
from recognize_anything.models.tag2text import ram


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
            checkpoint=checkpoints.grounded,
        )
        # Load tag2text model
        self.ram_model = self.load_ram_model(checkpoint=checkpoints.ram)
        # Load SAM predictor
        self.sam_predictor = SamPredictor(
            build_sam(checkpoint=checkpoints.sam).to(self.device)
        )

    def download_checkpoints(self):
        checkpoints_config = SLConfig.fromfile("./config/checkpoints.py")

        checkpoints = {
            "ram": checkpoints_config.ram_url,
            "grounded": checkpoints_config.grounded_url,
            "sam": checkpoints_config.sam_url,
        }

        checkpoints_dir = "checkpoints"
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        for name, url in checkpoints.items():
            checkpoint_filepath = os.path.join(
                checkpoints_dir, getattr(checkpoints_config, name)
            )
            if not os.path.exists(checkpoint_filepath):
                print(f"Downloading model: {name} from url: {url}")
                response = requests.get(url, stream=True)
                total_size_in_bytes = int(response.headers.get("content-length", 0))
                block_size = 1024  # 1 Kibibyte
                progress_bar = tqdm(
                    total=total_size_in_bytes, unit="iB", unit_scale=True
                )
                with open(checkpoint_filepath, "wb") as file:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        file.write(data)
                progress_bar.close()
                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    print("ERROR, something went wrong")

            # Update the attribute in the checkpoints_config
            setattr(checkpoints_config, name, checkpoint_filepath)

        return checkpoints_config

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

    def get_objects(self, image_filename):
        # Load image
        image_path = os.path.join("images", image_filename)
        image_pil = Image.open(image_path).convert("RGB")
        # Visualize raw image
        filename, ext = os.path.splitext(image_filename)
        raw_image_name = filename[: filename.rfind(".")] + "_raw" + ext
        image_pil.save(os.path.join(self.output_dir, raw_image_name))

        # Get tags using ram model
        image_raw = image_pil.resize((self.image_size, self.image_size))
        image_raw_tensor = self.normalize_image(image_pil=image_raw)
        image_with_batch = image_raw_tensor.unsqueeze(0).to(self.device)
        english_tags, chinese_tags = self.get_tags(image_with_batch)
        tags = english_tags.replace(" |", ",")
        print("Image Tags: ", tags)

        # Get boxes using grounding model
        image_tensor = self.normalize_image(image_pil=image_pil)
        boxes_filt, pred_phrases = self.get_boxes(
            image=image_tensor,
            image_pil=image_pil,
            tags=tags,
        )

        # Get masks using sam model
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = self.segmentate_image(image, boxes_filt)

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
        labeled_image_name = filename[: filename.rfind(".")] + "_labeled" + ext
        plt.savefig(
            os.path.join(self.output_dir, labeled_image_name),
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
        tags, chinese_tags = self.ram_model.generate_tag(image)

        return tags[0], chinese_tags[0]

    def segmentate_image(self, image, boxes_filt):
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

    def normalize_image(self, image_pil):
        normalize = TS.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = TS.Compose(
            [
                TS.Resize((self.image_size, self.image_size)),
                TS.ToTensor(),
                normalize,
            ]
        )
        image = transform(image_pil)
        return image

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
    masks, boxes, labels = segmentate_image.get_objects(
        image_filename="moveit_test_2.png"
    )


if __name__ == "__main__":
    main()
