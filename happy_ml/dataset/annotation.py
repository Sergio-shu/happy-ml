from typing import Dict, Union, Optional, List, Generic, TypeVar, Tuple, cast, Type
import abc

import numpy as np
import torch

A = TypeVar('A', bound='Annotation')
T = TypeVar('T', np.ndarray, torch.Tensor)
ScalarBoolType = Union[bool, T]


def collect_attribute_names(c: Type[A]) -> Tuple[str, ...]:
    slots = ()

    for super_c in c.__bases__:
        if issubclass(super_c, Annotation):
            slots += collect_attribute_names(super_c)
        else:
            continue

    slots += c.__slots__

    return slots


def collect_attribute_values(
        annotations: Union['Annotation[T]', List['Annotation[T]']]
) -> List[List[T]]:

    if isinstance(annotations, Annotation):
        annotations = [annotations]
        squeeze = True
    else:
        squeeze = False

    if len(annotations) == 0:
        return []

    annotation_class = type(annotations[0])
    attribute_names = collect_attribute_names(annotation_class)

    values = [
        [getattr(a, name) for a in annotations]
        for i, name in enumerate(attribute_names)
    ]
    if squeeze:
        values = [value[0] for value in values]

    return values


def collate_to_numpy(
        annotations: List[A],
        safe: bool = False,
) -> A:
    if len(annotations) == 0:
        raise ValueError('Annotation list must be non empty.')

    annotation_class = type(annotations[0])
    attributes_values = collect_attribute_values(annotations)

    if safe:
        all_values_are_numpy = all(
            [all([isinstance(value, np.ndarray) for value in attribute_values])
             for attribute_values in attributes_values]
        )
        if not all_values_are_numpy:
            raise ValueError('All values should be numpy.ndarray.')

    collated_values = [
        np.ascontiguousarray(np.concatenate(
            [value[None, :] for value in attribute_values]
        ))
        for attribute_values in attributes_values
    ]

    return annotation_class(*collated_values)


def collate_to_tensor(
        annotations: List[A],
        share: bool = False,
        safe: bool = False,
        clone: bool = False,
) -> A:
    if len(annotations) == 0:
        raise ValueError('Annotation list must be non empty.')

    annotation_class = type(annotations[0])

    if not issubclass(annotation_class, Annotation):
        raise ValueError(f'Collated objects should have Annotation base class.')

    attributes_values = collect_attribute_values(annotations)

    if safe:
        all_values_are_tensor = all(
            [all([isinstance(value, torch.Tensor) for value in attribute_values])
             for attribute_values in attributes_values]
        )
        if not all_values_are_tensor:
            raise ValueError('All values should be torch.Tensor.')

    collated_values = [
        torch.concatenate(
            [value[None, :].clone() for value in attribute_values]
        ).contiguous()
        for attribute_values in attributes_values
    ]
    if share:
        [tensor.share_memory_() for tensor in collated_values]

    return annotation_class(*collated_values)


def bool_as_numpy(value: Optional[Union[bool, T]]) -> T:
    if value is None:
        return np.array([False])

    if isinstance(value, bool):
        return np.asarray([value])

    return value


def int_as_numpy(value: Optional[Union[int, T]]) -> T:
    if value is None:
        return np.array([0])

    if isinstance(value, int):
        return np.asarray([value])

    return value


def random_bool_numpy() -> bool:
    return np.random.randint(0, 1) == 1


class Annotation(Generic[T]):
    __slots__ = ('annotated',)

    _defaults = {
        'annotated': False
    }

    def __init__(self, annotated: ScalarBoolType):
        self.annotated = bool_as_numpy(annotated)

    @property
    def is_annotated(self) -> T:
        return self.annotated

    def to_dict(self) -> Dict[str, T]:
        return {'annotated': self._annotated}

    def __eq__(self: 'Annotation[np.ndarray]', other: 'Annotation[np.ndarray]') -> bool:
        if type(other) is not type(self):
            raise ValueError('Objects being compared must be the same class.')

        values = collect_attribute_values(self)
        other_values = collect_attribute_values(other)

        return all(
            [np.array_equal(value, over_value)
             for value, over_value in zip(values, other_values)]
        )

    def __getitem__(self: A, index: int) -> A:
        values = collect_attribute_values(self)
        values_slice = [value[index] for value in values]

        return type(self)(*values_slice)

    def values_must_be_numpy(self):
        values = collect_attribute_values(self)

        if not all([isinstance(value, np.ndarray) for value in values]):
            raise ValueError('All values should be numpy.ndarray.')

    def values_must_be_tensor(self):
        values = collect_attribute_values(self)

        if not all([isinstance(value, torch.Tensor) for value in values]):
            raise ValueError('All values should be torch.Tensor.')

    def to_tensor(self, shared: bool = False) -> 'Annotation[torch.Tensor]':
        values = collect_attribute_values(self)
        tensors = [torch.Tensor(value) for value in values]
        if shared:
            [tensor.share_memory_() for tensor in tensors]

        return type(self)[torch.Tensor](*tensors)

    def to_numpy(self) -> 'Annotation[np.ndarray]':
        values = collect_attribute_values(self)
        converted_values = []

        for value in values:
            if isinstance(value, torch.Tensor):
                converted_values.append(value.numpy())
            elif isinstance(value, np.ndarray):
                converted_values.append(value)
            else:
                raise ValueError('Value should be numpy.ndarray or torch.Tensor')

        return type(self)(*converted_values)

    @classmethod
    def create_from_dict(cls, d: Dict[str, T]) -> 'Annotation[T]':
        return cls(**d)

    @classmethod
    @abc.abstractmethod
    def create_empty(cls) -> 'Annotation[T]':
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def create_random(cls) -> 'Annotation[T]':
        raise NotImplementedError()


DefaultImageShape = (224, 224, 3)


class ImageAnnotation(Annotation[T]):
    __slots__ = ('image_shape', 'image')

    default = {
        'image_shape': (224, 224, 3),
        'image': np.zeros((224, 224, 3), dtype=np.uint8),
    }

    # def __init__(
    #         self,
    #         annotated: ScalarBoolType,
    #         image_shape: Tuple[int, int, int],
    #         image: Optional[T],
    # ):
    #     super().__init__(annotated)
    #
    #     self._image_shape = np.array(image_shape) if isinstance(image_shape, Tuple) else image_shape
    #     self._image = image if image is not None else self.zero_image(image_shape)

    @property
    def image(self) -> Optional[T]:
        return self._image

    @staticmethod
    def zero_image(image_shape: Tuple[int, int, int]) -> T:
        return np.zeros(image_shape, dtype=np.uint8)

    @classmethod
    def create_empty(
            cls,
            image_shape: Tuple[int, int, int] = DefaultImageShape,
    ) -> 'ImageAnnotation[np.ndarray]':
        return cls(
            annotated=False,
            image_shape=image_shape,
            image=cls.zero_image(image_shape),
        )

    @classmethod
    def create_random(
            cls,
            image_shape: Tuple[int, int, int] = DefaultImageShape,
    ) -> 'ImageAnnotation[np.ndarray]':
        return cls(
            annotated=random_bool_numpy(),
            image_shape=image_shape,
            image=(np.random.random(image_shape) * 255).round().astype(np.float32),
        )

    def to_tensor(self, shared: bool = False) -> 'ImageAnnotation[torch.Tensor]':
        return cast(ImageAnnotation[torch.Tensor], super().to_tensor(shared))

    def to_numpy(self) -> 'ImageAnnotation[np.ndarray]':
        return cast(ImageAnnotation[np.ndarray], super().to_numpy())

    def to_dict(self) -> Dict[str, T]:
        d = super().to_dict()

        d['image_shape'] = self._image_shape
        d['image'] = self._image

        return d


class ExampleAnnotation(Annotation[T]):
    __slots__ = (
        '_vector_annotated',
        '_vector',
    )

    def __init__(
            self,
            annotated: ScalarBoolType,
            vector_annotated: ScalarBoolType,
            vector: Optional[T],
    ):
        super().__init__(annotated)

        self._vector_annotated = bool_as_numpy(vector_annotated)
        self._vector = vector if vector is not None else self.zero_vector()

    @property
    def is_vector_annotated(self) -> T:
        return self._vector_annotated

    @property
    def vector(self) -> T:
        return self._vector

    @staticmethod
    def zero_vector() -> T:
        return np.zeros((7, 2), dtype=np.float32)

    @classmethod
    def create_empty(cls) -> 'ExampleAnnotation[np.ndarray]':
        return ExampleAnnotation(
            annotated=False,
            vector_annotated=False,
            vector=cls.zero_vector(),
        )

    @classmethod
    def create_random(cls) -> 'ExampleAnnotation[np.ndarray]':
        return ExampleAnnotation(
            annotated=random_bool_numpy(),
            vector_annotated=random_bool_numpy(),
            vector=np.random.random((7, 2)).astype(np.float32),
        )

    def to_tensor(self, shared: bool = False) -> 'ExampleAnnotation[torch.Tensor]':
        return cast(ExampleAnnotation[torch.Tensor], super().to_tensor(shared))

    def to_numpy(self) -> 'ExampleAnnotation[np.ndarray]':
        return cast(ExampleAnnotation[np.ndarray], super().to_numpy())

    def to_dict(self) -> Dict[str, T]:
        d = super().to_dict()

        d['vector_annotated'] = self._vector_annotated
        d['vector'] = self._vector

        return d
