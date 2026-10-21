# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw


try:
    import cv2 as cv2
except Exception:
    # Pillow >= 9.1 moved these constants into enums; support both layouts.
    _AFFINE = getattr(getattr(Image, "Transform", Image), "AFFINE")
    _BILINEAR = getattr(getattr(Image, "Resampling", Image), "BILINEAR")

    class _CV2Fallback:
        COLOR_RGB2BGR = 4
        COLOR_BGR2RGB = 5
        INTER_LINEAR = 1
        INTER_AREA = 3
        LINE_AA = 16
        FONT_HERSHEY_SIMPLEX = 0

        def cvtColor(self, image, code):
            if code in (self.COLOR_RGB2BGR, self.COLOR_BGR2RGB):
                return image[..., ::-1].copy()
            raise NotImplementedError(f"Unsupported cvtColor code: {code}")

        def resize(self, image, dsize, interpolation=None):
            pil = Image.fromarray(image)
            return np.asarray(pil.resize(dsize, resample=_BILINEAR)).copy()

        def getAffineTransform(self, src, dst):
            src = np.asarray(src, dtype=np.float32)
            dst = np.asarray(dst, dtype=np.float32)
            a = np.concatenate([src, np.ones((3, 1), dtype=np.float32)], axis=1)
            matrix, _, _, _ = np.linalg.lstsq(a, dst, rcond=None)
            return matrix.T.astype(np.float32)

        def warpAffine(self, image, matrix, dsize, flags=None):
            width, height = dsize
            matrix3 = np.eye(3, dtype=np.float32)
            matrix3[:2] = np.asarray(matrix, dtype=np.float32)
            inverse = np.linalg.inv(matrix3)[:2].reshape(-1)

            squeeze_channel = image.ndim == 3 and image.shape[-1] == 1
            pil_input = image[..., 0] if squeeze_channel else image
            pil = Image.fromarray(pil_input)
            warped = pil.transform(
                (width, height),
                _AFFINE,
                tuple(float(x) for x in inverse),
                resample=_BILINEAR,
                fillcolor=0,
            )
            result = np.asarray(warped).copy()
            if not squeeze_channel and image.ndim == 3 and result.ndim == 2:
                result = result[:, :, None]
            return result

        def transform(self, points, matrix):
            points = np.asarray(points, dtype=np.float32)
            flat = points.reshape(-1, 2)
            homo = np.concatenate(
                [flat, np.ones((flat.shape[0], 1), dtype=np.float32)],
                axis=1,
            )
            transformed = homo @ np.asarray(matrix, dtype=np.float32).T
            return transformed.reshape(points.shape)

        def contourArea(self, points):
            points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
            x = points[:, 0]
            y = points[:, 1]
            return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

        def fillConvexPoly(self, image, points, color, lineType=None):
            pil = Image.fromarray(image)
            draw = ImageDraw.Draw(pil)
            xy = [tuple(map(int, pt)) for pt in np.asarray(points).reshape(-1, 2)]
            draw.polygon(xy, fill=tuple(int(c) for c in color))
            image[...] = np.asarray(pil)
            return image

        def addWeighted(self, src1, alpha, src2, beta, gamma):
            blended = np.clip(src1 * alpha + src2 * beta + gamma, 0, 255)
            return blended.astype(src1.dtype)

        def rectangle(self, image, pt1, pt2, color, thickness):
            pil = Image.fromarray(image)
            draw = ImageDraw.Draw(pil)
            if thickness < 0:
                draw.rectangle([pt1, pt2], fill=tuple(int(c) for c in color))
            else:
                draw.rectangle(
                    [pt1, pt2],
                    outline=tuple(int(c) for c in color),
                    width=thickness,
                )
            image[...] = np.asarray(pil)
            return image

        def putText(
            self,
            image,
            text,
            org,
            fontFace,
            fontScale,
            color,
            thickness,
            lineType=None,
        ):
            pil = Image.fromarray(image)
            draw = ImageDraw.Draw(pil)
            draw.text(org, text, fill=tuple(int(c) for c in color))
            image[...] = np.asarray(pil)
            return image

        def imwrite(self, filename, image):
            out = image
            if out.ndim == 3 and out.shape[-1] == 3:
                out = out[..., ::-1]
            Image.fromarray(out).save(filename)
            return True

        def imread(self, filename):
            image = np.asarray(Image.open(filename).convert("RGB"))
            return image[..., ::-1].copy()

    cv2 = _CV2Fallback()
