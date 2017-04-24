import numpy as np
from numpy.testing import assert_allclose

def assertIdenticalList(list1, list2):
	for i, list1i in enumerate(list1):
		if isinstance(list1i, bool):
			assert list1i is list2[i]
		else:
			assert_allclose(list1i, list2[i], rtol=1e-03)
