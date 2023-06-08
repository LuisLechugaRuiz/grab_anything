from rclpy.node import Node
import cv2
import os

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

import torch
import torchvision.transforms as TS

from PIL import Image


class SegmentateImageNode(Node):
    def __init__(self):
        self.declare_parameter("config", "")
        self.declare_parameter("tag2text_checkpoint", "")
        self.declare_parameter("grounded_checkpoint", "")
        self.declare_parameter("sam_checkpoint", "")
        self.declare_parameter("input_image", "")
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
        for logit, box in zip(logits_filt, boxes_filt):
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
        # load image
        image_pil, image = self.load_image(msg.data)
        # visualize raw image
        # TODO: Do we need this?
        image_pil.save(os.path.join(self.output_dir, "raw_image.jpg"))

        normalize = TS.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = TS.Compose([TS.Resize((384, 384)), TS.ToTensor(), normalize])
        raw_image = image_pil.resize((384, 384))
        raw_image = transform(raw_image).unsqueeze(0).to(self.device)

        specified_tags = "None"  # All by default.
        res = inference.inference(raw_image, self.tag2text_model, specified_tags)
        text_prompt = res[0].replace(" |", ",")
        caption = res[2]
        # TODO: Change by ros2 logs.
        print(f"Caption: {caption}")
        print(f"Tags: {text_prompt}")

        # run grounding dino model
        boxes_filt, scores, pred_phrases = self.get_grounding_output(
            model=self.groundino_model,
            image=image,
            caption=text_prompt,
        )

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

    # cfg
    config_file = args.config  # change the path of the model config file
    tag2text_checkpoint = args.tag2text_checkpoint  # change the path of the model
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_checkpoint = args.sam_checkpoint
    image_path = args.input_image
    split = args.split
    openai_key = args.openai_key
    openai_proxy = args.openai_proxy
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    iou_threshold = args.iou_threshold
    device = args.device

    # ChatGPT or nltk is required when using captions
    # openai.api_key = openai_key
    # if openai_proxy:
    # openai.proxy = {"http": openai_proxy, "https": openai_proxy}

    # initialize Tag2Text
    normalize = TS.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = TS.Compose([TS.Resize((384, 384)), TS.ToTensor(), normalize])

    # filter out attributes and action categories which are difficult to grounding
    delete_tag_index = []
    for i in range(3012, 3429):
        delete_tag_index.append(i)

    specified_tags = "None"
    # load model
    tag2text_model = tag2text.tag2text_caption(
        pretrained=tag2text_checkpoint,
        image_size=384,
        vit="swin_b",
        delete_tag_index=delete_tag_index,
    )
    # threshold for tagging
    # we reduce the threshold to obtain more tags
    tag2text_model.threshold = 0.64
    tag2text_model.eval()

    tag2text_model = tag2text_model.to(device)
    raw_image = image_pil.resize((384, 384))
    raw_image = transform(raw_image).unsqueeze(0).to(device)

    res = inference.inference(raw_image, tag2text_model, specified_tags)

    # Currently ", " is better for detecting single tags
    # while ". " is a little worse in some case
    text_prompt = res[0].replace(" |", ",")
    caption = res[2]

    print(f"Caption: {caption}")
    print(f"Tags: {text_prompt}")

    # run grounding dino model
    boxes_filt, scores, pred_phrases = get_grounding_output(
        model, image, text_prompt, box_threshold, text_threshold, device=device
    )

    # initialize SAM
    predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    # use NMS to handle overlapped boxes
    print(f"Before NMS: {boxes_filt.shape[0]} boxes")
    nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
    boxes_filt = boxes_filt[nms_idx]
    pred_phrases = [pred_phrases[idx] for idx in nms_idx]
    print(f"After NMS: {boxes_filt.shape[0]} boxes")
    caption = check_caption(caption, pred_phrases)
    print(f"Revise caption with number: {caption}")

    transformed_boxes = predictor.transform.apply_boxes_torch(
        boxes_filt, image.shape[:2]
    ).to(device)

    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes.to(device),
        multimask_output=False,
    )

    # draw output image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box, label in zip(boxes_filt, pred_phrases):
        show_box(box.numpy(), plt.gca(), label)

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
        os.path.join(output_dir, "automatic_label_output.jpg"),
        bbox_inches="tight",
        dpi=300,
        pad_inches=0.0,
    )

    save_mask_data(output_dir, caption, masks, boxes_filt, pred_phrases)
