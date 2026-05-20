"""Test PaddleOCR integration and version compatibility."""

from PIL import Image
import numpy as np
import pytest

# Skip entire module if optional ML dependencies aren't installed
pytest.importorskip("paddleocr")
pytest.importorskip("paddle")

from find_api.ml.ocr import OCRExtractor  # noqa: E402


@pytest.fixture
def ocr_extractor():
    """Initialize OCR extractor."""
    # Let initialization errors fail the test (don't silently skip)
    # since we already have pytest.importorskip guards at module level
    return OCRExtractor()


@pytest.fixture
def simple_image():
    """Create a simple test image (100x100 with white background)."""
    img = Image.new("RGB", (100, 100), color="white")
    return img


@pytest.fixture
def image_with_text():
    """Create an image with simple text using PIL ImageDraw."""
    from PIL import ImageDraw

    img = Image.new("RGB", (200, 100), color="white")
    draw = ImageDraw.Draw(img)
    # Draw simple black text
    draw.text((10, 10), "Hello", fill="black")
    return img


class TestOCRExtractor:
    """Test OCR functionality with PaddleOCR 2.7.3."""

    def test_extractor_initializes(self, ocr_extractor):
        """Test that OCRExtractor initializes without errors."""
        assert ocr_extractor is not None
        assert ocr_extractor.manager is not None

    def test_extract_text_returns_string(self, ocr_extractor, simple_image):
        """Test extract_text returns a string."""
        result = ocr_extractor.extract_text(simple_image)
        assert isinstance(result, str)

    def test_extract_text_accepts_pil_image(self, ocr_extractor, simple_image):
        """Test extract_text accepts PIL Image objects."""
        # Should not raise an exception
        result = ocr_extractor.extract_text(simple_image)
        assert isinstance(result, str)

    def test_extract_text_accepts_numpy_array(self, ocr_extractor, simple_image):
        """Test extract_text accepts numpy arrays."""
        image_array = np.array(simple_image)
        result = ocr_extractor.extract_text(image_array)
        assert isinstance(result, str)

    def test_extract_text_with_boxes_returns_list(self, ocr_extractor, simple_image):
        """Test extract_text_with_boxes returns a list of dicts."""
        result = ocr_extractor.extract_text_with_boxes(simple_image)
        assert isinstance(result, list)

    def test_extract_text_with_boxes_dict_structure(
        self, ocr_extractor, image_with_text
    ):
        """Test extract_text_with_boxes returns dicts with correct structure."""
        result = ocr_extractor.extract_text_with_boxes(image_with_text)

        # Result should be a list
        assert isinstance(result, list)

        # With our test image containing text, we should get at least one result
        # (May be empty if model doesn't detect the simple PIL-drawn text,
        # so we validate structure if any results exist)
        for item in result:
            assert isinstance(item, dict)
            assert "text" in item
            assert "confidence" in item
            assert "bbox" in item
            assert isinstance(item["text"], str)
            assert isinstance(item["confidence"], float)

            # bbox should have coordinates
            bbox = item["bbox"]
            assert "x1" in bbox
            assert "y1" in bbox
            assert "x2" in bbox
            assert "y2" in bbox
            assert all(isinstance(v, (int, float)) for v in bbox.values())

    @pytest.mark.slow
    def test_paddleocr_api_compatibility(self, ocr_extractor, image_with_text):
        """Test that PaddleOCR 2.7.3 is usable through the public API.

        Marked as slow because this exercises real PaddleOCR initialization,
        which may download model weights on first run (slow/non-hermetic).
        Skip in fast test runs with: pytest -m "not slow"
        """
        result = ocr_extractor.extract_text(image_with_text)
        assert isinstance(result, str)
        # Verify OCR actually extracted text (not empty)
        assert len(result) > 0, "OCR should extract text from image with text"


class TestOCRErrorHandling:
    """Test error handling in OCR extraction."""

    def test_extract_text_handles_invalid_image(self, ocr_extractor):
        """Test that invalid very small images propagate OCR errors."""
        # Very small image (1x1) will likely cause PaddleOCR to fail
        # The implementation logs and re-raises, so we expect an exception
        img = Image.new("RGB", (1, 1))
        with pytest.raises(Exception):
            ocr_extractor.extract_text(img)
