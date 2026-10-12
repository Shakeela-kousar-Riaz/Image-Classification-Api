
import io

from PIL import Image, ImageOps, UnidentifiedImageError
from fastapi import HTTPException, UploadFile

MAX_FILE_SIZE = 5 * 1024 * 1024
MAX_PIXELS = 20_000_000
ALLOWED_TYPES = {"image/jpeg", "image/png"}


async def read_and_validate_image(file: UploadFile) -> Image.Image:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Only JPEG and PNG images are supported."
        )

    content = await file.read(MAX_FILE_SIZE + 1)

    if not content:
        raise HTTPException(status_code=400, detail="Empty file.")

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Maximum file size is 5 MB."
        )

    try:
        with Image.open(io.BytesIO(content)) as image:
            if image.format not in {"JPEG", "PNG"}:
                raise HTTPException(
                    status_code=415,
                    detail="Image content must be JPEG or PNG."
                )

            if image.width * image.height > MAX_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail="Image dimensions are too large."
                )

            image.verify()

        with Image.open(io.BytesIO(content)) as image:
            image = ImageOps.exif_transpose(image)
            return image.convert("RGB")

    except HTTPException:
        raise
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image."
        )
