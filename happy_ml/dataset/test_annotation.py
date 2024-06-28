import unittest

import numpy as np

from happy_ml.dataset import annotation


class AnnotationBaseTestCases(unittest.TestCase):

    def __init__(self, methodName=...):
        super().__init__(methodName)

        self.annotation_classes = [
            annotation.ImageAnnotation,
            annotation.ExampleAnnotation,
        ]

    def test_all_numpy(self):
        for annotation_class in self.annotation_classes:
            a = annotation_class.create_random()
            a.values_must_be_numpy()

    def test_all_tensor(self):
        for annotation_class in self.annotation_classes:
            a = annotation_class.create_random().to_tensor()
            a.values_must_be_tensor()

    def test_dict_format(self):
        for annotation_class in self.annotation_classes:
            a = annotation_class.create_random()
            a_dict = a.to_dict()
            a_from_dict = annotation_class.create_from_dict(a_dict)
            assert a == a_from_dict

    def test_collation_to_numpy(self):
        number_to_collate = 10

        for annotation_class in self.annotation_classes:
            # create random annotations and collate them
            annotations = [
                annotation_class.create_random()
                for _ in range(number_to_collate)
            ]
            collated_to_numpy = annotation.collate_to_numpy(annotations, safe=True)
            AnnotationBaseTestCases._check_collated_values(annotations, collated_to_numpy, number_to_collate)

    def test_collation_to_tensor(self):
        number_to_collate = 10

        for annotation_class in self.annotation_classes:
            # create random annotations and collate them
            annotations = [
                annotation_class.create_random().to_tensor()
                for _ in range(number_to_collate)
            ]
            collated = annotation.collate_to_tensor(annotations, safe=True)

            # check each attribute value to be equal
            AnnotationBaseTestCases._check_collated_values(annotations, collated, number_to_collate)

    @staticmethod
    def _check_collated_values(annotations, collated_to_numpy, number_to_collate):
        all_attributes = annotation.collect_attribute_names(type(collated_to_numpy))
        for attribute_name in all_attributes:
            assert len(getattr(collated_to_numpy, attribute_name)) == number_to_collate
            for i, a in enumerate(annotations):
                assert np.array_equal(getattr(a, attribute_name), getattr(collated_to_numpy, attribute_name)[i])
