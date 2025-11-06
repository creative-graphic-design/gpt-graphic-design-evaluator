import base64
import io

from PIL import Image


def image_to_base64(image: Image.Image) -> str:
    """Convert a PIL Image to a base64-encoded PNG string.

    Args:
        image (Image.Image): The input PIL Image.

    Returns:
        str: The base64-encoded PNG string.
    """
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str
