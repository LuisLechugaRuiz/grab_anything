import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

# Ros
from rclpy.node import Node

# Torch
import torch
import torchvision
import torchvision.transforms as TS

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import (
    clean_state_dict,
    get_phrases_from_posmap,
)

# Segmentate
from segment_anything import build_sam, SamPredictor

# Tag2Text
from Tag2Text.models import tag2text
from Tag2Text import inference


class SegmentateImageNode(Node):
    def __init__(self):
        self.declare_parameter("config", "")
        self.declare_parameter("tag2text_checkpoint", "")
        self.declare_parameter("grounded_checkpoint", "")
        self.declare_parameter("sam_checkpoint", "")
        self.declare_parameter("split", ",")
        self.declare_parameter("openai_key", "")
        self.declare_parameter("openai_proxy", None)
        self.declare_parameter("output_dir", "outputs")
        self.declare_parameter("box_threshold", 0.25)
        self.declare_parameter("text_threshold", 0.2)
        self.declare_parameter("iou_threshold", 0.5)
        self.declare_parameter("device", "cpu")
        self.declare_parameter("output_dir", "")
        # TODO: add output folder as a parameter.
        # Create output dir if it does not exist.
        self.output_dir = self.get_parameter("output_dir").value
        os.makedirs(self.output_dir, exist_ok=True)
        config_file = self.get_parameter("config").value
        self.device = self.get_parameter("device").value
        self.box_threshold = self.get_parameter("box_threshold").value
        self.text_threshold = self.get_parameter("text_threshold").value
        self.iou_threshold = self.get_parameter("iou_threshold").value

        # Load grounding model
        grounded_checkpoint = self.get_parameter("grounded_checkpoint").value
        self.groundino_model = self.load_groundino_model(
            config_file, grounded_checkpoint
        )

        # Load tag2text model
        tag2text_checkpoint = self.get_parameter("tag2text_checkpoint").value
        self.tag2text_model = self.load_tag2text_model(
            tag2text_checkpoint=tag2text_checkpoint
        )

        # Load SAM predictor
        sam_checkpoint = self.get_parameter("sam_checkpoint").value
        self.sam_predictor = SamPredictor(
            build_sam(checkpoint=sam_checkpoint).to(self.device)
        )

    def get_grounding_output(self, model, image, caption):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        model = model.to(self.device)
        image = image.to(self.device)
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
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
        tokenized = tokenlizer(caption)
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

    def load_groundino_model(self, model_config_path, model_checkpoint_path):
        args = SLConfig.fromfile(model_config_path)
        args.device = self.device
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        load_res = model.load_state_dict(
            clean_state_dict(checkpoint["model"]), strict=False
        )
        print(load_res)
        _ = model.eval()
        return model

    def load_tag2text_model(self, tag2text_checkpoint):
        # filter out attributes and action categories which are difficult to grounding
        delete_tag_index = []
        for i in range(3012, 3429):
            delete_tag_index.append(i)

        # load model
        model = tag2text.tag2text_caption(
            pretrained=tag2text_checkpoint,
            image_size=384,
            vit="swin_b",
            delete_tag_index=delete_tag_index,
        )
        # threshold for tagging
        # we reduce the threshold to obtain more tags
        model.threshold = 0.64
        model.eval()

        return model.to(self.device)

    # TODO: Make a ros2 subscriber to send the new image.
    def on_new_image(self, msg):
        # Using msg.data as image_path to avoid creating ros msg.
        image_path = msg.data

        # load image
        image_pil, image = self.load_image(image_path)
        # visualize raw image
        # TODO: Do we need this?
        image_pil.save(os.path.join(self.output_dir, "raw_image.jpg"))

        normalize = TS.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = TS.Compose([TS.Resize((384, 384)), TS.ToTensor(), normalize])
        raw_image = image_pil.resize((384, 384))
        raw_image = transform(raw_image).unsqueeze(0).to(self.device)

        # get labels using tag2text model
        specified_tags = "None"  # All by default.
        res = inference.inference(raw_image, self.tag2text_model, specified_tags)
        text_prompt = res[0].replace(" |", ",")
        caption = res[2]
        # TODO: Change by ros2 logs.
        print(f"Caption: {caption}")
        print(f"Tags: {text_prompt}")

        # get boxes using grounding model
        boxes_filt, pred_phrases = self.get_boxes(
            image=image,
            image_pil=image_pil,
            text_prompt=text_prompt,
            caption=caption,
        )

        # get masks using sam model
        masks = self.segmentate_image(image_path, boxes_filt)

        # draw output image
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        for mask in masks:
            self.show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        for box, label in zip(boxes_filt, pred_phrases):
            self.show_box(box.numpy(), plt.gca(), label)

        plt.title(
            "Tag2Text-Captioning: "
            + caption
            + "\n"
            + "Tag2Text-Tagging"
            + text_prompt
            + "\n"
        )
        plt.axis("off")
        plt.savefig(
            os.path.join(self.output_dir, "automatic_label_output.jpg"),
            bbox_inches="tight",
            dpi=300,
            pad_inches=0.0,
        )

        self.save_mask_data(self.output_dir, caption, masks, boxes_filt, pred_phrases)

    def get_boxes(self, image, image_pil, text_prompt, caption):
        boxes_filt, scores, pred_phrases = self.get_grounding_output(
            model=self.groundino_model,
            image=image,
            caption=text_prompt,
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
        print(f"Revise caption with number: {caption}")
        return boxes_filt, pred_phrases

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
        # load image
        image_pil = Image.open(image_path).convert("RGB")  # load image

        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w
        return image_pil, image

    def save_mask_data(output_dir, caption, mask_list, box_list, label_list):
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
            "caption": caption,
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

    def show_box(box, ax, label):
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
