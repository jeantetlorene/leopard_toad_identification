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

        return T.functional.pad(image, padding, fill=self.fill)


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
                # 1. Fix Aspect Ratio & Pad
                ResizeAndPad(size, fill=0),
                # 2. Safe Geometric Augmentations
                T.Pad(padding=int(size * 0.1)),
                T.RandomCrop(size=size),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=15),
                # 3. Photometric Augmentations (IR Simulation)
                T.RandomApply(
                    [
                        T.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                T.RandomGrayscale(p=0.2),
                # 4. Blur (Simulating focus issues)
                T.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                # 5. Normalization
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __call__(self, x):
        return self.transform(x), self.transform(x)
