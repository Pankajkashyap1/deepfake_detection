import torch
import torch.nn.functional as F


def token_gradcam(model, image_tensor, target_class=None):
    """
    Transformer-style Grad-CAM using gradients of the target logit
    with respect to the input patch tokens of the reasoning transformer.
    """
    model.eval()
    model.zero_grad()

    outputs = model(image_tensor)

    logits = outputs["logits"]                 # (B,2)
    pre_reason_tokens = outputs["pre_reason_tokens"]  # (B,393,768)

    if target_class is None:
        target_class = outputs["preds"]        # (B,)
    elif isinstance(target_class, int):
        target_class = torch.full(
            (logits.size(0),),
            target_class,
            device=logits.device,
            dtype=torch.long
        )

    score = logits.gather(1, target_class.unsqueeze(1)).sum()

    grads = torch.autograd.grad(
        outputs=score,
        inputs=pre_reason_tokens,
        retain_graph=False,
        create_graph=False,
        allow_unused=False
    )[0]                                       # (B,393,768)

    # patch tokens only: token 0 is global
    acts = pre_reason_tokens[:, 1:197, :]      # (B,196,768)
    grads = grads[:, 1:197, :]                 # (B,196,768)

    # standard Grad-CAM style: channel importance
    weights = grads.mean(dim=1, keepdim=True)  # (B,1,768)

    cam = (acts * weights).sum(dim=-1)         # (B,196)
    cam = F.relu(cam)

    B = cam.size(0)
    cam = cam.view(B, 1, 14, 14)

    cam = F.interpolate(
        cam,
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    ).squeeze(1)

    cam_min = cam.amin(dim=(1, 2), keepdim=True)
    cam_max = cam.amax(dim=(1, 2), keepdim=True)
    cam = (cam - cam_min) / (cam_max - cam_min + 1e-6)

    return cam.detach(), outputs