def build_custom_aug():
    import albumentations as A

    return [
        # resolution
        A.AdditiveNoise(p=0.05),
        A.Downscale(p=0.05),
        A.Sharpen(p=0.05),
        A.MotionBlur(p=0.05),
        A.Defocus(p=0.05),

        # brightness, contrast, color
        A.CLAHE(p=0.05),
        A.RandomGamma(p=0.05),
        A.PlasmaBrightnessContrast(p=0.05),
        A.HueSaturationValue(p=0.05),
        A.ToGray(p=0.05),

        # weather
        A.RandomFog(p=0.05),
        A.RandomRain(p=0.05),
        A.RandomSunFlare(p=0.05),
        A.RandomShadow(p=0.05),
        A.Spatter(p=0.05),

        # geometric
        A.Perspective(scale=(0.02, 0.05), p=0.08),
        A.Affine(translate_percent=0.1, scale=(0.7, 1.3), shear=(-10, 10), p=0.08),
        A.Rotate(limit=15, p=0.08),
    ]