import unittest
import cv2
from bound_box import predict_image, mask_outside_area, get_bound_area
from Draw_bound_Vid import draw_rectangle_with_label

class TestPredictImage(unittest.TestCase):
  def setUp(self):
    # Set up any data that is needed before running the tests
    self.image = cv2.imread('test_image.jpg')
    self.Threshold = 0.5

  def test_predict_image_returns_correct_output(self):
    # Test that the predict_image() function returns the expected output
    score, prediction = predict_image(self.image, self.Threshold)
    self.assertEqual(score, 0.9)
    self.assertEqual(prediction, 'pikachu')

  def test_predict_image_handles_invalid_inputs(self):
    # Test that the predict_image() function can handle invalid inputs
    score, prediction = predict_image(None, self.Threshold)
    self.assertEqual(score, 0)
    self.assertEqual(prediction, 'none')

class TestDrawRectangleWithLabel(unittest.TestCase):
  def setUp(self):
    # Set up any data that is needed before running the tests
    self.image = cv2.imread('test_image.jpg')
    self.label = 'pikachu'
    self.x = 10
    self.y = 10
    self.width = 50
    self.height = 50
    self.color = (255, 0, 0)

  def test_draw_rectangle_with_label_returns_correct_output(self):
    # Test that the draw_rectangle_with_label() function returns the expected output
    result = draw_rectangle_with_label(self.image, self.label, self.x, self.y, self.width, self.height, self.color)
    self.assertIsNotNone(result)

class TestMaskOutsideArea(unittest.TestCase):
  def setUp(self):
    # Set up any data that is needed before running the tests
    self.image = cv2.imread('test_image.jpg')
    self.x1 = 10
    self.y1 = 10
    self.x2 = 50
    self.y2 = 50

  def test_mask_outside_area_returns_correct_output(self):
    # Test that the mask_outside_area() function returns the expected output
    result = mask_outside_area(self.image, self.x1, self.y1, self.x2, self.y2)
    self.assertIsNotNone(result)

class TestGetBoundArea(unittest.TestCase):
  def setUp(self):
    # Set up any data that is needed before running the tests
    self.image = cv2.imread('test_image.jpg')
    self.class_to_look_for = 'pikachu'
    self.threshold = 0.5

  def test_get_bound_area_returns_correct_output(self):
    # Test that the get_bound_area() function returns the expected output
    result = get_bound_area(self.image, self.class_to_look_for, self.threshold)
    self.assertIsNotNone(result)