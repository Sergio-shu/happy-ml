from typing import Optional, List, Type


from happy_ml.dataset import annotation


class AnnotatedSample:
    __slots__ = ('_annotations', '_annotations_by_type')

    def __init__(self, annotations: List[annotation.Annotation]):
        self._annotations = annotations
        self._annotations_by_type = {
            type(a).__name__: a for a in annotations
        }

    @property
    def annotations(self) -> List[annotation.Annotation]:
        return self._annotations

    def has_annotation(self, t: Type[annotation.A]) -> bool:
        return t.__name__ in self._annotations_by_type

    def get_annotation(self, t: Type[annotation.A]) -> Optional[annotation.A]:
        if t.__name__ in self._annotations_by_type:
            return self._annotations_by_type[t.__name__]
        return None
