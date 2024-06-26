import unittest

import torch
from torch import multiprocessing
import numpy as np

from happy_ml.dataset import annotation


class AnnotationSharingTestCases(unittest.TestCase):

    @staticmethod
    def test_shared_numpy_data_source():
        number_to_collate = 10
        annotations = [
            annotation.ExampleAnnotation.create_random()
            for _ in range(number_to_collate)
        ]
        collated = annotation.collate_to_numpy(annotations)

        a = collated[0]
        b = collated[-1]

        collated.vector[0, :] = 1
        assert np.array_equal(collated.vector[0], a.vector)

        b.vector[:] = 2
        assert np.array_equal(collated.vector[-1], b.vector)

    @staticmethod
    def test_shared_tensor_data_source():
        number_to_collate = 10
        annotations = [
            annotation.ExampleAnnotation.create_random().to_tensor()
            for _ in range(number_to_collate)
        ]
        collated = annotation.collate_to_tensor(annotations)

        a = collated[0]
        b = collated[-1]

        collated.vector[0, :] = 1
        assert np.array_equal(collated.vector[0], a.vector)

        b.vector[:] = 2
        assert np.array_equal(collated.vector[-1], b.vector)

    @staticmethod
    def test_shared_tensor_data_source_in_multiprocessing():
        number_to_collate = 10
        annotations = [
            annotation.ExampleAnnotation.create_random().to_tensor()
            for _ in range(number_to_collate)
        ]
        data_source = annotation.collate_to_tensor(annotations, safe=True, share=True)
        local_ptr = data_source.vector.data_ptr()

        queue = multiprocessing.Queue()

        def _worker_write(_, ds: annotation.ExampleAnnotation[torch.Tensor]):
            a = ds.vector[0]
            a[:] = 1
            b = ds[1]
            b.vector[:] = 1
            queue.put(ds.vector.data_ptr())

        ctx = multiprocessing.start_processes(
            _worker_write,
            args=(data_source,),
            nprocs=1,
            join=False,
            daemon=True,
            start_method='fork',
        )
        worker_ptr = queue.get()
        ctx.join()
        assert local_ptr == worker_ptr
        assert torch.equal(data_source.vector[0], torch.ones_like(data_source.vector[0]))
        assert torch.equal(data_source.vector[1], torch.ones_like(data_source.vector[1]))
        assert not torch.equal(data_source.vector[2], torch.ones_like(data_source.vector[2]))

    @staticmethod
    def test_shared_numpy_data_source_in_multiprocessing_not_working():
        number_to_collate = 10
        annotations = [
            annotation.ExampleAnnotation.create_random()
            for _ in range(number_to_collate)
        ]
        data_source = annotation.collate_to_numpy(annotations, safe=True)

        def _worker_write(_, ds: annotation.ExampleAnnotation[np.ndarray]):
            a = ds.vector[0]
            a[:] = 1
            # data ptr from ds.keypoints.__array_interface__['data'][0]
            # could be the same, but object is not actually shared

        multiprocessing.start_processes(
            _worker_write,
            args=(data_source,),
            nprocs=1,
            join=True,
            daemon=True,
            start_method='fork',
        )

        assert not np.array_equal(data_source.vector[0], np.ones_like(data_source.vector[0]))
