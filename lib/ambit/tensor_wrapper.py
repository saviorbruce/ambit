from . import pyambit
import numbers


class LabeledTensorProduct:
    def __init__(self, left, right):
        self.tensors = []
        self.tensors.append(left)
        self.tensors.append(right)

    def __mul__(self, other):
        if isinstance(other, LabeledTensor):
            self.tensors.append(other)
            return self
        else:
            print("LabeledTensorProduct::mul %s not implemented" % (type(other)))
            return NotImplemented

    def __float__(self):
        if len(self.tensors) != 2:
            raise RuntimeError("Conversion operator only supports binary expressions.")

        R = Tensor(self.tensors[0].tensor.type, "R", [])
        R.contract(self.tensors[0], self.tensors[1], [], self.tensors[0].indices, self.tensors[1].indices,
                   self.tensors[0].factor * self.tensors[1].factor, 1.0)

        C = Tensor(pyambit.TensorType.kCore, "C", [])
        C.slice(R, [], [])

        return C.data()[0]


class LabeledTensorAddition:
    def __init__(self, left, right):
        self.tensors = []
        self.tensors.append(left)
        self.tensors.append(right)

    def __mul__(self, other):
        if isinstance(other, LabeledTensor):
            return LabeledTensorDistributive(other, self)
        elif isinstance(other, numbers.Number):
            for tensor in self.tensors:
                tensor.factor *= other
            return self

    def __neg__(self):
        for tensor in self.tensors:
            tensor.factor *= -1.0
        return self

    def __rmul__(self, other):
        if isinstance(other, numbers.Number):
            for tensor in self.tensors:
                tensor.factor *= other
            return self

    def __add__(self, other):
        self.tensors.append(other)
        return self


class LabeledTensorDistributive:
    def __init__(self, left, right):
        self.A = left
        self.B = right

    def __float__(self):
        R = Tensor(self.A.tensor.type, "R", [])

        for tensor in self.B.tensors:
            R.contract(self.A, tensor, [], self.A.indices, tensor.indices, self.A.factor * tensor.factor, 1.0)

        C = Tensor(pyambit.TensorType.kCore, "C", [])
        C.slice(R, [], [])

        return C.data()[0]


class LabeledTensor:
    def __init__(self, t, indices):
        self.factor = 1.0
        self.tensor = t
        self.indices = pyambit.Indices.split(indices)
        self.labeledTensor = pyambit.LabeledTensor(self.tensor, self.indices, self.factor)

    def __neg__(self):
        self.factor *= -1.0
        return self

    def __mul__(self, other):
        if isinstance(other, numbers.Number):
            self.factor *= other
            return self
        elif isinstance(other, LabeledTensorAddition):
            return LabeledTensorDistributive(self, other)
        else:
            return LabeledTensorProduct(self, other)

    def __add__(self, other):
        return LabeledTensorAddition(self, other)

    def __sub__(self, other):
        other.factor *= -1.0
        return LabeledTensorAddition(self, other)

    def __rmul__(self, other):
        if isinstance(other, numbers.Number):
            self.factor *= other

            return self
        else:
            return NotImplemented

    def __iadd__(self, other):
        if isinstance(other, LabeledTensor):
            self.tensor.permute(other.tensor, self.indices, other.indices, other.factor, self.factor)
            return None
        elif isinstance(other, LabeledTensorDistributive):
            pass
        elif isinstance(other, LabeledTensorProduct):
            # Only support pairwise for now
            if len(other.tensors) != 2:
                raise RuntimeError("LabeledTensor: __imul__ : Only pairwise contractions are supported.")

            A = other.tensors[0]
            B = other.tensors[1]

            self.tensor.contract(A.tensor, B.tensor, self.indices, A.indices, B.indices, A.factor * B.factor,
                                 self.factor)

            # This operator is complete.
            return None

        elif isinstance(other, LabeledTensorAddition):
            pass
        else:
            print("LabeledTensor::__iadd__ not implemented for this type.")
            return NotImplemented

    def __isub__(self, other):
        if isinstance(other, LabeledTensor):
            self.tensor.permute(other.tensor, self.indices, other.indices, -other.factor, self.factor)
            return None
        elif isinstance(other, LabeledTensorDistributive):
            pass
        elif isinstance(other, LabeledTensorProduct):
            # Only support pairwise for now
            if len(other.tensors) != 2:
                raise RuntimeError("LabeledTensor: __imul__ : Only pairwise contractions are supported.")

            A = other.tensors[0]
            B = other.tensors[1]

            self.tensor.contract(A.tensor, B.tensor, self.indices, A.indices, B.indices, -A.factor * B.factor,
                                 self.factor)

            # This operator is complete.
            return None

        elif isinstance(other, LabeledTensorAddition):
            pass
        else:
            print("LabeledTensor::__isub__ not implemented for this type.")
            return NotImplemented

    def __imul__(self, other):
        if isinstance(other, numbers.Number):
            self.tensor.scale(other)
            return None

        else:
            return NotImplemented

    def __itruediv__(self, other):
        if isinstance(other, numbers.Number):
            self.tensor.scale(1.0 / other)
            return None
        else:
            return NotImplemented


class Tensor:
    @staticmethod
    def build(type, name, dims):
        return Tensor(type, name, dims)

    def __init__(self, type=None, name=None, dims=None, existing=None):
        if existing:
            self.tensor = existing
            self.type = existing.type
            self.dims = existing.dims
            self.name = name if name else existing.name
        else:
            self.name = name
            self.rank = len(dims)
            self.type = type
            self.dims = dims
            self.tensor = pyambit.ITensor.build(type, name, dims)

    def __getitem__(self, indices):
        if isinstance(indices, list):
            return SlicedTensor(self, indices)
        else:
            return LabeledTensor(self.tensor, indices)

    def __setitem__(self, indices_str, value):

        if isinstance(value, SlicedTensor):
            if self.tensor == value.tensor:
                raise RuntimeError("SlicedTensor::__setitem__: Self assignment is not allowed.")
            if self.tensor.rank != value.tensor.rank:
                raise RuntimeError("SlicedTensor::__setitem__: Sliced tensors do not have same rank")

            self.tensor.slice(value.tensor.tensor, indices_str, value.range, value.factor, 0.0)

            return None

        indices = pyambit.Indices.split(str(indices_str))

        if isinstance(value, LabeledTensorProduct):
            # This is "simple assignment" C = A * B

            if len(value.tensors) != 2:
                raise ArithmeticError("Only pair-wise contractions are currently supported")

            A = value.tensors[0]
            B = value.tensors[1]

            self.tensor.contract(A.tensor, B.tensor, indices, A.indices, B.indices, A.factor * B.factor, 0.0)

        elif isinstance(value, LabeledTensorAddition):
            self.tensor.zero()

            for tensor in value.tensors:
                if isinstance(tensor, LabeledTensor):
                    self.tensor.permute(tensor.tensor, indices, tensor.indices, tensor.factor, 1.0)
                else:
                    # recursively call set item
                    self.factor = 1.0
                    self.__setitem__(indices_str, tensor)

        # This should be handled by LabeledTensor above
        elif isinstance(value, LabeledTensor):

            if self == value.tensor:
                raise RuntimeError("Self-assignment is not allowed.")
            if self.tensor.rank != value.tensor.rank:
                raise ArithmeticError("Permuted tensors do not have same rank")

            self.tensor.permute(value.tensor, indices, value.indices, value.factor, 0.0)

        elif isinstance(value, LabeledTensorDistributive):

            self.tensor.zero()

            A = value.A
            for B in value.B.tensors:
                self.tensor.contract(A.tensor, B.tensor, indices, A.indices, B.indices, A.factor * B.factor, 1.0)

    def __eq__(self, other):
        if isinstance(other, Tensor):
            return self.tensor == other.tensor
        elif isinstance(other, pyambit.ITensor):
            return self.tensor == other
        else:
            return NotImplemented

    def printf(self):
        self.tensor.printf()

    def data(self):
        return self.tensor.data()

    def norm(self, type):
        return self.tensor.norm(type)

    def zero(self):
        self.tensor.zero()

    def scale(self, beta):
        self.tensor.scale(beta)

    def copy(self, other):
        self.tensor.copy(other)

    def slice(self, A, Cinds, Ainds, alpha=1.0, beta=0.0):
        self.tensor.slice(A.tensor, Cinds, Ainds, alpha, beta)

    def permute(self, A, Cinds, Ainds, alpha=1.0, beta=0.0):
        self.tensor.permute(A.tensor, Cinds, Ainds, alpha, beta)

    def contract(self, A, B, Cinds, Ainds, Binds, alpha=1.0, beta=0.0):
        self.tensor.contract(A.tensor, B.tensor, Cinds, Ainds, Binds, alpha, beta)

    def gemm(self, A, B, transA, transB, nrow, ncol, nzip, ldaA, ldaB, ldaC, offA=0, offB=0, offC=0, alpha=1.0,
             beta=0.0):
        self.tensor.gemm(A.tensor, B.tensor, transA, transB, nrow, ncol, nzip, ldaA, ldaB, ldaC, offA, offB, offC,
                         alpha, beta)

    def syev(self, order):
        aResults = self.tensor.syev(order)

        results = {}
        for k, v in aResults.iteritems():
            results[k] = Tensor(existing=v)

        return results

    def power(self, p, condition=1.0e-12):
        aResult = self.tensor.power(p, condition)
        return Tensor(existing=aResult)


class SlicedTensor:
    def __init__(self, tensor, range, factor=1.0):
        self.tensor = tensor
        self.range = range
        self.factor = factor

        # Check the data given to us
        if not isinstance(tensor, Tensor):
            raise RuntimeError("SlicedTensor: Expected tensor to be Tensor")

        if len(range) != tensor.rank:
            raise RuntimeError(
                "SlicedTensor: Sliced tensor does not have correct number of indices for underlying tensor's rank")

        for idx, value in enumerate(range):
            if len(value) != 2:
                raise RuntimeError(
                    "SlicedTensor: Each index of an IndexRange should have two elements {start,end+1} in it.")
            if value[0] > value[1]:
                raise RuntimeError("SlicedTensor: Each index of an IndexRange should end+1>=start in it.")
            if value[1] > tensor.dims[idx]:
                raise RuntimeError("SlicedTensor: IndexRange exceeds size of tensor.")

    def __iadd__(self, value):
        if isinstance(value, SlicedTensor):
            if self.tensor == value.tensor:
                raise RuntimeError("SlicedTensor::__setitem__: Self assignment is not allowed.")
            if self.tensor.rank != value.tensor.rank:
                raise RuntimeError("SlicedTensor::__setitem__: Sliced tensors do not have same rank")

            self.tensor.slice(value.tensor, self.range, value.range, value.factor, 1.0)

            return None

    def __isub__(self, value):
        if isinstance(value, SlicedTensor):
            if self.tensor == value.tensor:
                raise RuntimeError("SlicedTensor::__setitem__: Self assignment is not allowed.")
            if self.tensor.rank != value.tensor.rank:
                raise RuntimeError("SlicedTensor::__setitem__: Sliced tensors do not have same rank")

            self.tensor.slice(value.tensor, self.range, value.range, -value.factor, 1.0)

            return None

    def __mul__(self, other):
        if isinstance(other, numbers.Number):
            return SlicedTensor(self.tensor, self.range, other * self.factor)

    def __rmul__(self, other):
        if isinstance(other, numbers.Number):
            return SlicedTensor(self.tensor, self.range, other * self.factor)

    def __neg__(self):
        return SlicedTensor(self.tensor, self.range, -self.factor)