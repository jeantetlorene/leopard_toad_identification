import random
import torchvision.transforms as T
from PIL import Image, ImageFilter


class ResizeAndPad:
    """
    Resizes the image to target_size while keeping aspect ratio,
    then pads the shorter side with black (0) to make it a square.
    Prevents "squashing" of the leopard patterns.
    """

    def __init__(self, target_size, fill=0):
        self.target_size = target_size
        self.fill = fill

    def __call__(self, image):
        w, h = image.size
        scale = self.target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)

        image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)

        delta_w, delta_h = self.target_size - new_w, self.target_size - new_h
        padding = (
            delta_w // 2,
            delta_h // 2,
            delta_w - (delta_w // 2),
            delta_h - (delta_h // 2),
        )

        # Use reflection padding to avoid artificial black borders while keeping aspect ratio
        return T.functional.pad(image, padding, padding_mode="reflect")


class GaussianBlur:
    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return x.filter(ImageFilter.GaussianBlur(radius=sigma))


class SimCLRTransform:
    def __init__(self, size):
        self.transform = T.Compose(
            [
                # 1. Aspect Ratio Preserving Resize with Reflection Padding
                ResizeAndPad(size),
                # 2. Geometric Augmentations (Preserving Full Pattern)
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=15),
                # RandomAffine provides translation and slight scaling without aggressive cropping
                T.RandomAffine(
                    degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5
                ),
                # 3. Photometric Augmentations
                T.RandomApply(
                    [
                        T.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                T.RandomGrayscale(p=0.2),
                # 4. Blur
                T.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                # 5. Normalization
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __call__(self, x):
        return self.transform(x), self.transform(x)


def get_inference_transform(size):
    """
    Standard inference transform:
    1. Aspect Ratio Preserving Resize with Reflection Padding
    2. Conversion to Tensor
    3. Normalization (ImageNet stats)
    """
    return T.Compose(
        [
            ResizeAndPad(size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
