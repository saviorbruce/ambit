#if !defined(TENSOR_INCLUDE_TENSOR_H)
#define TENSOR_INCLUDE_TENSOR_H

#include <cstdio>
#include <utility>
#include <vector>
#include <map>
#include <string>
#include <functional>

#include <memory>
#include <tuple>
#include <stdexcept>

namespace ambit {

using std::tuple;
using std::shared_ptr;
using std::unique_ptr;

static constexpr double numerical_zero__ = 1.0e-15;

// => Forward Declarations <=
class TensorImpl;
class LabeledTensor;
class LabeledTensorProduct;
class LabeledTensorAddition;
class LabeledTensorSubtraction;
class LabeledTensorDistributive;
class LabeledTensorSumOfProducts;
class SlicedTensor;

// => Tensor Types <=
enum TensorType {
    kCurrent,     // <= If cloning from existing tensor use its type.
    kCore,        // <= In-core only tensor
    kDisk,        // <= Disk cachable tensor
    kDistributed, // <= Tensor suitable for parallel distributed
    kAgnostic,    // <= Let the library decide for you.
    kGlobalArray  // <= Use the Global Array Framework
};

enum EigenvalueOrder {
    kAscending, kDescending
};

// => Typedefs <=
typedef std::vector<size_t> Dimension;
typedef std::vector<std::vector<size_t>> IndexRange;
typedef std::vector<std::string> Indices;

// => Settings Namespace <=
namespace settings {

/** Number of MPI processes.
 *
 * For single process runs this will always be 1.
 */
extern int nprocess;

/// Rank of this process. (zero-based)
extern int rank;

/// Print debug information? true, or false
extern bool debug;

/// Memory usage limit. Default is 1GB.
extern size_t memory_limit;

/// Distributed capable?
extern const bool distributed_capable;

/// Enable timers
extern bool timers;

}

/** Initializes the tensor library.
 *
 * Calls any necessary initialization of utilized frameworks.
 * @param argc number of command line arguments
 * @param argv the command line arguments
 * @return error code
 */
int initialize(int argc = 0, char* * argv = nullptr);

/** Shutdowns the tensor library.
 *
 * Calls any necessary routines of utilized frameworks.
 */
void finalize();

/** Barrier function
 *
 * If called in MPI process it equivalent to calling MPI_Barrier.
 * Otherwise this call is nop.
 */
void barrier();

class Tensor {

public:

    // => Constructors <= //

    /**
     * Factory constructor. Builds a Tensor of TensorType type
     * with given name and dimensions dims
     *
     * Parameters:
     *  @param type the TensorType to build
     *  @param name the name of the Tensor
     *  @param dims the dimensions of the indices of the tensor
     *    (dims.size() is the tensor rank)
     *
     * Results:
     *  @return new Tensor of TensorType type with name and dims
     *   The returned Tensor is set to zero.
     **/
    static Tensor build(
        TensorType type,
        const std::string& name,
        const Dimension& dims);

    /**
     * Return a new Tensor of TensorType type which copies the name,
     * dimensions, and data of this tensor.
     *
     * E.g.:
     *  Tensor A = C.clone(kDisk);
     * is equivalent to:
     *  Tensor A = Tensor::build(kDisk, C.name(), C.dims());
     *  A->copy(C);
     *
     * Parameters:
     *  @param type the TensorType to use for the clone
     *
     * Results:
     *  @return new Tensor of TensorType type with the name and contents of this
     **/
    Tensor clone(TensorType type = kCurrent) const;

    /**
     * Default constructor, builds a Tensor with a null underlying
     * implementation.
     *
     * Calling any methods of such a Tensor will result in exceptions being
     * thrown.
     **/
    Tensor();

    /**
     * Frees the Tensor's internal memory allocation. This is for users
     * that want finer control on memory consumption. After calling this
     * function the Tensor is no longer valid and cannot be used in
     * furthur calls.
     */
    void reset();

    // => Accessors <= //

    /// @return The tensor type enum, one of kCore, kDisk, kDistributed
    TensorType type() const;
    /// @return The name of the tensor for use in printing
    std::string name() const;
    /// @return The dimension of each index in the tensor
    const Dimension& dims() const;
    /// @return The dimension of the ind-th index
    size_t dim(size_t ind) const;
    /// @return The number of indices in the tensor
    size_t rank() const;
    /// @return The total number of elements in the tensor (product of dims)
    size_t numel() const;

    /// Set the name of the tensor to name
    void set_name(const std::string& name);

    /// @return Does this Tensor point to the same underlying tensor as Tensor other?
    bool operator==(const Tensor& other) const;
    /// @return !Does this Tensor point to the same underlying tensor as Tensor other?
    bool operator!=(const Tensor& other) const;

    /**
     * Print some tensor information to fh
     * \param level If level = false, just print name and dimensions.  If level = true, print the entire tensor.
     **/
    void print(FILE* fh = stdout, bool level = true, const std::string& format = std::string("%11.6f"), int maxcols = 5) const;

    // => Data Access <= //

    /**
     * Returns the raw data vector underlying the tensor object if the
     * underlying tensor object supports a raw data vector. This is only the
     * case if the underlying tensor is of type kCore.
     *
     * This routine is intended to facilitate rapid filling of data into a
     * kCore buffer tensor, following which the user may stripe the buffer
     * tensor into a kDisk or kDistributed tensor via slice operations.
     *
     * If a vector is successfully returned, it points to the unrolled data of
     * the tensor, with the right-most dimensions running fastest and left-most
     * dimensions running slowest.
     *
     * Example successful use case:
     *  Tensor A = Tensor::build(kCore, "A3", {4,5,6});
     *  std::vector<double>& Av = A.data();
     *  double* Ap = Av.data(); // In case the raw pointer is needed
     *  In this case, Av[0] = A(0,0,0), Av[1] = A(0,0,1), etc.
     *
     *  Tensor B = Tensor::build(kDisk, "B3", {4,5,6});
     *  std::vector<double>& Bv = B.data(); // throws
     *
     * Results:
     *  @return data pointer, if tensor object supports it
     **/
    std::vector<double>& data();
    const std::vector<double>& data() const;

    // => BLAS-Type Tensor Operations <= //

    /**
     * Returns the norm of the tensor
     *
     * Parameters:
     * @param type the type of norm desired:
     *  0 - Infinity-norm, maximum absolute value of elements
     *  1 - One-norm, sum of absolute values of elements
     *  2 - Two-norm, square root of sum of squares
     *
     * Results:
     *  @return computed norm
     **/
    double norm(int type = 2) const;

    /** Find the maximum value.
     *
     * @return maximum value along with its indices
     */
    std::tuple<double, std::vector<size_t>> max() const;

    /** Find the minimum value.
     *
     * @return minimum value along with its indices
     */
    std::tuple<double, std::vector<size_t>> min() const;

    /**
     * Sets the data of the tensor to zeros
     *  C = 0.0
     *
     * Note: this just drops down to scale(0.0);
     *
     * Results:
     *  C is the current tensor, whose data is overwritten
     **/
    void zero();

    /**
     * Scales the tensor by scalar beta, e.g.:
     *  C = beta * C
     *
     * Note: If beta is 0.0, a memset is performed rather than a scale to clamp
     * NaNs and other garbage out.
     *
     * Parameters:
     *  @param beta the scale to apply
     *
     * Results:
     *  C is the current tensor, whose data is overwritten
     **/
    void scale(double beta = 0.0);

    /**
     * Sets all elements in the tensor to the value.
     *
     * @param alpha the value to set
     */
    void set(double alpha);

    /**
     * Copy the data of other into this tensor:
     *  C() = other()
     * Note: this just drops into slice
     *
     * Parameters:
     *  @param other the tensor to copy data from
     *
     * Results
     *  C is the current tensor, whose data is overwritten
     **/
    void copy(const Tensor& other);

    /**
     * Perform the slice:
     *  C(Cinds) = alpha * A(Ainds) + beta * C(Cinds)
     *
     * Note: Most users should instead use the operator overloading
     * routines, e.g.,
     *  C2({{0,m},{0,n}}) += 0.5 * A2({{1,m+1},{1,n+1}});
     *
     * Parameters:
     *  @param A The source tensor, e.g., A2
     *  @param Cinds The slices of indices of tensor C, e.g., {{0,m},{0,n}}
     *  @param Ainds The indices of tensor A, e.g., {{1,m+1},{1,n+1}}
     *  @param alpha The scale applied to the tensor A, e.g., 0.5
     *  @param beta The scale applied to the tensor C, e.g., 1.0
     *
     * Results:
     *  C is the current tensor, whose data is overwritten. e.g., C2
     *  All elements outside of the IndexRange in C are untouched, alpha and beta
     *  scales are applied only to elements indices of the IndexRange
     **/
    void slice(
        const Tensor& A,
        const IndexRange& Cinds,
        const IndexRange& Ainds,
        double alpha = 1.0,
        double beta = 0.0);

    /**
     * Perform the permutation:
     *  C(Cinds) = alpha * A(Ainds) + beta * C(Cinds)
     *
     * Note: Most users should instead use the operator overloading
     * routines, e.g.,
     *  C2("ij") += 0.5 * A2("ji");
     *
     * Parameters:
     *  @param A The source tensor, e.g., A2
     *  @param Cinds The indices of tensor C, e.g., "ij"
     *  @param Ainds The indices of tensor A, e.g., "ji"
     *  @param alpha The scale applied to the tensor A, e.g., 0.5
     *  @param beta The scale applied to the tensor C, e.g., 1.0
     *
     * Results:
     *  C is the current tensor, whose data is overwritten. e.g., C2
     **/
    void permute(
        const Tensor& A,
        const Indices& Cinds,
        const Indices& Ainds,
        double alpha = 1.0,
        double beta = 0.0);

    /**
     * Perform the contraction:
     *  C(Cinds) = alpha * A(Ainds) * B(Binds) + beta * C(Cinds)
     *
     * Note: Most users should instead use the operator overloading
     * routines, e.g.,
     *  C2("ij") += 0.5 * A2("ik") * B2("jk");
     *
     * Parameters:
     *  @param A The left-side factor tensor, e.g., A2
     *  @param B The right-side factor tensor, e.g., B2
     *  @param Cinds The indices of tensor C, e.g., "ij"
     *  @param Ainds The indices of tensor A, e.g., "ik"
     *  @param Binds The indices of tensor B, e.g., "jk"
     *  @param alpha The scale applied to the product A*B, e.g., 0.5
     *  @param beta The scale applied to the tensor C, e.g., 1.0
     *
     * Results:
     *  C is the current tensor, whose data is overwritten. e.g., C2
     **/
    void contract(
        const Tensor& A,
        const Tensor& B,
        const Indices& Cinds,
        const Indices& Ainds,
        const Indices& Binds,
        double alpha = 1.0,
        double beta = 0.0);

    /**
     * Perform the GEMM call equivalent to:
     *  C_DGEMM(
     *      (transA ? 'T' : 'N'),
     *      (transB ? 'T' : 'N'),
     *      nrow,
     *      ncol,
     *      nzip,
     *      alpha,
     *      Ap + offA,
     *      ldaA,
     *      Bp + offB,
     *      ldaB,
     *      beta,
     *      Cp + offC,
     *      ldaC);
     *  where, e.g., Ap = A.data().data();
     *
     * Notes:
     *  - This is only implemented for kCore
     *  - No bounds checking on the GEMM is performed
     *  - This function is intended to help advanced users get optimal
     *    performance from single-node codes.
     *
     * Parameters:
     *  @param A the left-side factor tensor
     *  @param B the right-side factor tensor
     *  @param transA transpose A or not
     *  @param transB transpose B or not
     *  @param nrow number of rows in the GEMM call
     *  @param ncol number of columns in the GEMM call
     *  @param nzip number of zip indices in the GEMM call
     *  @param ldaA leading dimension of A:
     *   Must be >= nzip if transA == false
     *   Must be >= nrow if transA == true
     *  @param ldaB leading dimension of B:
     *   Must be >= ncol if transB == false
     *   Must be >= nzip if transB == true
     *  @param ldaC leading dimension of C:
     *   Must be >= ncol
     *  @param offA the offset of the A data pointer to apply
     *  @param offB the offset of the B data pointer to apply
     *  @param offC the offset of the C data pointer to apply
     *  @param alpha the scale to apply to A*B
     *  @param beta the scale to apply to C
     *
     * Results:
     *  C is the current tensor, whose data is overwritten.
     *  All elements in C outside of the range traversed by gemm are
     *  untouched.
     **/
    void gemm(
        const Tensor& A,
        const Tensor& B,
        bool transA,
        bool transB,
        size_t nrow,
        size_t ncol,
        size_t nzip,
        size_t ldaA,
        size_t ldaB,
        size_t ldaC,
        size_t offA = 0L,
        size_t offB = 0L,
        size_t offC = 0L,
        double alpha = 1.0,
        double beta = 0.0);

    // => Rank-2 LAPACK-Type Tensor Operations <= //

    /**
     * This routine computes all the eigenvalues and eigenvectors of
     * a square real symmetric matrix (this, A).
     *
     * The eigenvector v(j) of this satifies the following formula:
     *
     * A*v(j) = lambda(j)*v(j)
     *
     * where
     *
     * lambda(j) is its eigenvalue.
     *
     * The computed eigenvectors are orthonormal.
     *
     * @returns map of Tensor with the keys "eigenvalues" and "eigenvectors".
     */
    std::map<std::string, Tensor> syev(EigenvalueOrder order) const;
    Tensor power(double power, double condition = 1.0E-12) const;

    /**
     * This routine computes a square real general matrix (this, A), the
     * eigenvalues and the left and right eigenvectors.
     *
     * The right eigenvector v(j) of A satisfies the following formula:
     *
     * A*v(j) = lambda(j)*v(j)
     *
     * where
     *
     * lambda(j) is its eigenvalue.
     *
     * The left eigenvector u(j) of A satisfies the following formula:
     *
     * u(j)H*A = lambda(j)*u(j)H
     *
     * where
     *
     * u(j)H denotes the conjugate transpose of u(j).
     *
     * The computed eigenvectors are normalized so that their Euclidean
     * norm equals one and the largest component is real.
     *
     * @returns map of Tensor with the keys "lambda", "lambda i", "v", and "u". See
     * definitions above.
     */
    std::map<std::string, Tensor> geev(EigenvalueOrder order) const;
    //std::map<std::string, Tensor> svd() const;

    //void potrf();
    //void potri();
    //void potrs(const Tensor& L);
    //void posv(const Tensor& A);

    //void trtrs(const Tensor& L,

    //void getrf();
    //void getri();
    //void getrs(const Tensor& LU);
    //void gesv(const Tensor& A);

    //std::map<std::string, Tensor> lu() const;
    //std::map<std::string, Tensor> qr() const;

    //Tensor inverse() const;

    void test_function();



    // => Utility Operations <= //

    static Tensor cat(const std::vector<Tensor>, int dim);

    // => Iterators <= //

    void iterate(const std::function<void (const std::vector<size_t>&, double&)>& func);
    void citerate(const std::function<void (const std::vector<size_t>&, const double&)>& func) const;

private:

    shared_ptr<TensorImpl> tensor_;

protected:

    Tensor(shared_ptr<TensorImpl> tensor);

    static std::map<std::string, Tensor> map_to_tensor(const std::map<std::string, TensorImpl*>& x);

public:

    // => Operator Overloading API <= //

    LabeledTensor operator()(const std::string& indices) const;

    SlicedTensor operator()(const IndexRange& range) const;
    SlicedTensor operator()() const;

    // => Environment <= //

private:

    static std::string scratch_path__;

public:

    static void set_scratch_path(const std::string& path) { scratch_path__ = path; }
    static std::string scratch_path() { return scratch_path__; }

};

class LabeledTensor {

public:
    LabeledTensor(Tensor T, const Indices& indices, double factor = 1.0);

    double factor() const { return factor_; }
    const Indices& indices() const { return indices_; }
    Indices& indices() { return indices_; }
    const Tensor& T() const { return T_; }

    LabeledTensorProduct operator*(const LabeledTensor& rhs);
    LabeledTensorAddition operator+(const LabeledTensor& rhs);
    LabeledTensorAddition operator-(const LabeledTensor& rhs);

    LabeledTensorDistributive operator*(const LabeledTensorAddition& rhs);

    /** Copies data from rhs to this sorting the data if needed. */
    void operator=(const LabeledTensor& rhs);
    void operator+=(const LabeledTensor& rhs);
    void operator-=(const LabeledTensor& rhs);

    void operator=(const LabeledTensorDistributive& rhs);
    void operator+=(const LabeledTensorDistributive& rhs);
    void operator-=(const LabeledTensorDistributive& rhs);

    void operator=(const LabeledTensorProduct& rhs);
    void operator+=(const LabeledTensorProduct& rhs);
    void operator-=(const LabeledTensorProduct& rhs);

    void operator=(const LabeledTensorAddition& rhs);
    void operator+=(const LabeledTensorAddition& rhs);
    void operator-=(const LabeledTensorAddition& rhs);

    void operator*=(double scale);
    void operator/=(double scale);

//    bool operator==(const LabeledTensor& other) const;
//    bool operator!=(const LabeledTensor& other) const;

    size_t numdim() const { return indices_.size(); }
    size_t dim_by_index(const std::string& idx) const;

    // negation
    LabeledTensor operator-() const {
        return LabeledTensor(T_, indices_, -factor_);
    }

    void contract(const LabeledTensorProduct& rhs, bool zero_result, bool add);

private:

    void set(const LabeledTensor& to);

    Tensor T_;
    Indices indices_;
    double factor_;

};

inline LabeledTensor operator*(double factor, const LabeledTensor& ti) {
    return LabeledTensor(ti.T(), ti.indices(), factor*ti.factor());
};

class LabeledTensorProduct {

public:
    LabeledTensorProduct(const LabeledTensor& A, const LabeledTensor& B)
    {
        tensors_.push_back(A);
        tensors_.push_back(B);
    }

    LabeledTensorProduct() {}

    size_t size() const { return tensors_.size(); }

    const LabeledTensor& operator[](size_t i) const { return tensors_[i]; }

    LabeledTensorProduct& operator*(const LabeledTensor& other) {
        tensors_.push_back(other);
        return *this;
    }

    void operator*=(const LabeledTensor& other) {
        tensors_.push_back(other);
    }

    // conversion operator
    operator double() const;

    std::pair<double, double> compute_contraction_cost(const std::vector<size_t>& perm) const;

private:

    std::vector<LabeledTensor> tensors_;
};

class LabeledTensorAddition
{
public:
    LabeledTensorAddition(const LabeledTensor& A, const LabeledTensor& B)
    {
        tensors_.push_back(A);
        tensors_.push_back(B);
    }

    size_t size() const { return tensors_.size(); }

    const LabeledTensor& operator[](size_t i) const { return tensors_[i]; }

    std::vector<LabeledTensor>::iterator begin() { return tensors_.begin(); }
    std::vector<LabeledTensor>::const_iterator begin() const { return tensors_.begin(); }

    std::vector<LabeledTensor>::iterator end() { return tensors_.end(); }
    std::vector<LabeledTensor>::const_iterator end() const { return tensors_.end(); }

    LabeledTensorAddition& operator+(const LabeledTensor& other) {
        tensors_.push_back(other);
        return *this;
    }

    LabeledTensorAddition& operator-(const LabeledTensor& other) {
        tensors_.push_back(-other);
        return *this;
    }

    LabeledTensorDistributive operator*(const LabeledTensor& other);

    LabeledTensorAddition& operator*(double scalar);

    // negation
    LabeledTensorAddition& operator-();

private:

    // This handles cases like T("ijab")
    std::vector<LabeledTensor> tensors_;

};

inline LabeledTensorAddition operator*(double factor, const LabeledTensorAddition& ti) {
    LabeledTensorAddition ti2 = ti;
    return ti2 * factor;
}

// Is responsible for expressions like D * (J - K) --> D*J - D*K
class LabeledTensorDistributive
{
public:
    LabeledTensorDistributive(const LabeledTensor& A, const LabeledTensorAddition& B)
            : A_(A), B_(B)
    {}

    const LabeledTensor& A() const { return A_; }
    const LabeledTensorAddition& B() const { return B_; }

    // conversion operator
    operator double() const;

private:

    const LabeledTensor& A_;
    const LabeledTensorAddition& B_;

};

class SlicedTensor
{
public:
    SlicedTensor(Tensor T, const IndexRange& range, double factor = 1.0);

    double factor() const { return factor_; }
    const IndexRange& range() const { return range_; }
    const Tensor& T() const { return T_; }

    void operator=(const SlicedTensor& rhs);
    void operator+=(const SlicedTensor& rhs);
    void operator-=(const SlicedTensor& rhs);

    // negation
    SlicedTensor operator-() const {
        return SlicedTensor(T_, range_, -factor_);
    }
private:
    Tensor T_;
    IndexRange range_;
    double factor_;
};

inline SlicedTensor operator*(double factor, const SlicedTensor& ti) {
    return SlicedTensor(ti.T(), ti.range(), factor*ti.factor());
};

}

#endif

