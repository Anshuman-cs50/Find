#!/usr/bin/env python3
"""
Manual OCR verification script - run with: python scripts/manual_ocr_check.py

Tests PaddleOCR 2.7.3 API compatibility and basic functionality.
This is a standalone utility, not a pytest test.
"""

import sys
from pathlib import Path


def check_version():
    """Verify PaddleOCR version."""
    import paddleocr

    version = paddleocr.__version__
    print(f"✓ PaddleOCR version: {version}")
    # Must be exactly 2.7.3 as pinned in pyproject.toml
    if version != "2.7.3":
        raise AssertionError(
            f"Expected PaddleOCR 2.7.3 (pinned version), got {version}"
        )


def check_ocr_extractor(OCRExtractor, Image, np):
    """Test OCR extractor initialization and basic functionality."""
    print("\n--- Testing OCRExtractor ---")

    try:
        extractor = OCRExtractor()
        print("✓ OCRExtractor initialized")
    except Exception as e:
        print(f"✗ Failed to initialize OCRExtractor: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Create test image
    test_image = Image.new("RGB", (100, 100), color="white")
    print("✓ Created test image")

    # Test extract_text with PIL Image
    try:
        result = extractor.extract_text(test_image)
        print(f"✓ extract_text(PIL Image) returned: {type(result).__name__}")
        assert isinstance(result, str), "extract_text should return string"
    except Exception as e:
        print(f"✗ extract_text failed: {e}")
        return False

    # Test extract_text with numpy array
    try:
        result = extractor.extract_text(np.array(test_image))
        print(f"✓ extract_text(numpy array) returned: {type(result).__name__}")
        assert isinstance(result, str), "extract_text should return string"
    except Exception as e:
        print(f"✗ extract_text with numpy failed: {e}")
        return False

    # Test extract_text_with_boxes
    try:
        result = extractor.extract_text_with_boxes(test_image)
        print(f"✓ extract_text_with_boxes returned: {type(result).__name__}")
        assert isinstance(result, list), "extract_text_with_boxes should return list"
        print(f"  Extracted {len(result)} text regions")
    except Exception as e:
        print(f"✗ extract_text_with_boxes failed: {e}")
        return False

    return True


if __name__ == "__main__":
    # Defer heavy imports to avoid issues during pytest collection
    from PIL import Image
    import numpy as np

    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from find_api.ml.ocr import OCRExtractor

    print("=" * 50)
    print("PaddleOCR 2.7.3 Compatibility Test")
    print("=" * 50)

    try:
        check_version()
        success = check_ocr_extractor(OCRExtractor, Image, np)

        if success:
            print("\n" + "=" * 50)
            print("✓ All tests passed! PaddleOCR 2.7.3 is working")
            print("=" * 50)
            sys.exit(0)
        else:
            print("\n✗ Some tests failed")
            sys.exit(1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
