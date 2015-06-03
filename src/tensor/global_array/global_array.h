#if !defined(GLOBAL_ARRAY_H)
#define GLOBAL_ARRAY_H

#include "tensor/tensorimpl.h"
#include "ga.h"
#include "macdecls.h"

namespace ambit {

namespace global_array {

int initialize(int argc, char* * argv);

//int initialize(MPI_Comm comm, int argc = 0, char * * argv = nullptr);

void finalize();

class GlobalArrayImpl : public TensorImpl
{
public:
    GlobalArrayImpl(const std::string& name, const Dimension& dims);
    ~GlobalArrayImpl();

    int global_array() const { return global_array_; }

    // in case this is a "fake" tensor with only a single value
    double single_value() const { return single_value_; }

    // => Simple Single Tensor Operations <= //

    double norm(
            int type = 2) const;

    std::tuple<double, std::vector<size_t>> max() const;

    std::tuple<double, std::vector<size_t>> min() const;

    void scale(double beta = 0.0);

    void set(double alpha);

    void permute(
            ConstTensorImplPtr A,
            const std::vector<std::string>& Cinds,
            const std::vector<std::string>& Ainds,
            double alpha = 1.0,
            double beta = 0.0);

    void contract(
            ConstTensorImplPtr A,
            ConstTensorImplPtr B,
            const std::vector<std::string>& Cinds,
            const std::vector<std::string>& Ainds,
            const std::vector<std::string>& Binds,
            double alpha = 1.0,
            double beta = 0.0);

//    // => Order-2 Operations <= //

    std::map<std::string, TensorImplPtr> syev(EigenvalueOrder order) const;

    TensorImplPtr power(double alpha, double condition) const;

    void iterate(const std::function<void (const std::vector<size_t>&, double&)>& func);
    void citerate(const std::function<void (const std::vector<size_t>&, const double&)>& func) const;

private:
    int global_array_;
    int ga_rank = 0;
    int* ga_dims;
    int* ga_chunk;
    char* ga_name = "GA";
    double single_value_ = 0;
};

}

typedef global_array::GlobalArrayImpl* GlobalArrayImplPtr;
typedef const global_array::GlobalArrayImpl* ConstGlobalArrayImplPtr;

}

#endif
