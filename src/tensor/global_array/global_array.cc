#if !defined(HAVE_GA)
#error The Global Array interface is being compiled without Global Array Library present.
#endif

#include "global_array.h"
#include "../globals.h"
#include "math/math.h"
#include <ambit/print.h>
#include <math.h>
#include <cmath>
#include <ambit/tensor.h>
#include <ambit/timer.h>
#include "tensor/indices.h"

#include "../core/core.h"

namespace ambit {
namespace global_array {

// did we initialize MPI or did the user?
int initialized_mpi = 0;


int initialize(int argc, char **argv)
{
    // check if MPI is already initialized
    MPI_Initialized(&initialized_mpi);
    if (!initialized_mpi) {
        int error = MPI_Init(&argc, &argv);
        if (error != MPI_SUCCESS) {
            throw std::runtime_error("global_array::initialize: Unable to initialize MPI.");
        }
    }

    // Initialize GA
    GA_Initialize();

    settings::rank = GA_Nodeid();
    settings::nprocess = GA_Nnodes();

    if (settings::debug) {
        print("Global Array interface initialized. nprocess: %d\n", settings::nprocess);
    }

    return 0;
}

void finalize()
{
    GA_Terminate();
    // if we initialized MPI then we finalize it.
    MPI_Initialized(&initialized_mpi);
    if (initialized_mpi)    MPI_Finalize();
}

GlobalArrayImpl::GlobalArrayImpl(const std::string &name,
                                     const Dimension &dims)
        : TensorImpl(kGlobalArray, name, dims)
{
    if (dims.size() == 0) {
        global_array_ = 0;
        return;
    }

    ga_rank = dims.size();
    ga_dims = new int[ga_rank];
//    ga_chunk = nullptr;
    ga_chunk = new int[ga_rank];
//    ga_name = new char[name.size()+1];
//    std::copy(name.begin(),name.end(),ga_name);
//    ga_name[name.size()] = '\0';
    for(int i=0; i<ga_rank; ++i) {
        ga_dims[i] = dims[i];
        ga_chunk[i] = -1;
    }
    // prevent the last dimension to be divided?
//    ga_chunk[ga_rank-1] = 1;
    global_array_ = NGA_Create(C_DBL,ga_rank,ga_dims,ga_name,ga_chunk);
    if (global_array_ == 0) {
        char err_msg[] = "Global Array Creation Failed! dim=";
        GA_Error(err_msg, ga_rank);
    }
    GA_Zero(global_array_);
}

GlobalArrayImpl::~GlobalArrayImpl()
{
    if (ga_rank == 0) return;
    GA_Destroy(global_array_);
}

double GlobalArrayImpl::norm(int type) const
{
    if (ga_rank == 0) return fabs(single_value_);
    double nm = 0.0;
    switch (type) {
        case 0: // Infinity-norm, find the maximum absolute value
            citerate([&](const std::vector<size_t> &indices, const double &value) {
                if (nm<fabs(value)) nm=fabs(value);
            });
            GA_Dgop(&nm,1,"absmax");
            break;
        case 1: // One-norm, sum of absolute values
            citerate([&](const std::vector<size_t> &indices, const double &value) {
                nm += fabs(value);
            });
            GA_Dgop(&nm,1,"+");
            break;
        case 2: // Two-norm, square root of sum of squares
            nm = GA_Ddot(global_array_,global_array_);
            nm = std::sqrt(nm);
            break;
        default:
            throw std::runtime_error("Unknown norm type passed to Global Array.");
    }
    return nm;
}


std::tuple<double, std::vector<size_t>> GlobalArrayImpl::max() const
{
    std::tuple<double, std::vector<size_t>> result;

    if (ga_rank == 0) {
        std::get<0>(result) = fabs(single_value_);
        std::get<1>(result) = {0};
    }
    else {
        double element_value;
        std::vector<size_t> element_indices(ga_rank);
        int index[ga_rank];
        NGA_Select_elem(global_array_,"max",&element_value,index);
        for(int n=0; n<ga_rank; ++n) element_indices[n] = index[n];

        std::get<0>(result) = element_value;
        std::get<1>(result) = element_indices;
    }

    return result;
}
std::tuple<double, std::vector<size_t>> GlobalArrayImpl::min() const
{
    std::tuple<double, std::vector<size_t>> result;

    if (ga_rank == 0) {
        std::get<0>(result) = fabs(single_value_);
        std::get<1>(result) = {0};
    }
    else {
        double element_value;
        std::vector<size_t> element_indices(ga_rank);
        int index[ga_rank];
        NGA_Select_elem(global_array_,"min",&element_value,index);
        for(int n=0; n<ga_rank; ++n) element_indices[n] = index[n];

        std::get<0>(result) = element_value;
        std::get<1>(result) = element_indices;
    }

    return result;
}

void GlobalArrayImpl::scale(double beta)
{
    if (ga_rank == 0) {
        single_value_ *= beta;
        return;
    }
    if (beta == 0.0) {
        GA_Zero(global_array_);
    }
    else if(beta == 1.0) {
        return;
    }
    else {
        GA_Scale(global_array_, &beta);
    }
}

void GlobalArrayImpl::set(double alpha)
{
    if (ga_rank == 0) {
        single_value_ = alpha;
        return;
    }
    GA_Fill(global_array_,&alpha);
}

void GlobalArrayImpl::permute(
        ConstTensorImplPtr A,
        const std::vector<std::string> &Cinds,
        const std::vector<std::string> &Ainds,
        double alpha,
        double beta)
{
//    ambit::timer::timer_push("P:");
    if (ga_rank != 0) {

        const GlobalArrayImpl* cA = dynamic_cast<const GlobalArrayImpl*>((A));
        int tA = cA->global_array_;


        std::vector<int> same(ga_rank, 0);
        int total_same = 0;
        for(int ind=0; ind < Cinds.size(); ++ind )
            if(Cinds[ind].compare(Ainds[ind]) == 0)
            {
                same[ind] = 1;
                total_same++;
            }
        // In case the indices are all same, no need to permute
        if(total_same == ga_rank) {
            GA_Add(&alpha,tA,&beta,global_array_,global_array_);
            return;
        }
        else {
            // do the scale first
            scale(beta);
            // Check the number of Cinds and Ainds
            if(Cinds.size()!=ga_rank || Ainds.size()!= ga_rank)
                throw std::runtime_error("The number of indices is inconsistent with Tensor rank!");

            // Create a permutation map (all indices should be named differently)
            std::vector<int> permutation_map(ga_rank);
            for(int i=0; i<ga_rank; ++i) {
                for (int j=0; j<ga_rank; ++j) {
                    if(Cinds[i].compare(Ainds[j]) == 0)
                        permutation_map[i] = j;
                }
            }
            // Check the dimensions after permutation
            for(int i=0; i<ga_rank; ++i) {
                if(ga_dims[i] != A->dims()[permutation_map[i]]) {
                    throw std::runtime_error("Permuted tensors do not have same dimensions!\n");
                }
            }
            // Find out what is on processor "me"
            int me = settings::rank;
            int lo[ga_rank],hi[ga_rank];
            NGA_Distribution(global_array_,me,lo,hi);
            // check if there is actually data on me
            if (lo[0] < 0 && hi[0] < 0) {
                GA_Sync();
                return;
            }
            // The dimensions for local core tensors
            Dimension dims_A_part(ga_rank);
            Dimension dims_C_part(ga_rank);
            // Determine which part of A's data I need
            std::vector<int> loA(ga_rank);
            std::vector<int> hiA(ga_rank);
            std::vector<int> ldA(ga_rank);
            int nelem = 1;
            for(int ind=0; ind<ga_rank; ++ind) {
                int indA = permutation_map[ind];
                int length = hi[ind] - lo[ind] + 1;
                loA[indA] = lo[ind];
                hiA[indA] = hi[ind];
                ldA[indA] = length;
                nelem *= length;
                dims_A_part[indA] = length;
                dims_C_part[ind] = length;
            }
            ldA.erase(ldA.begin());
            // Get the target data from A
            std::vector<double> buffA(nelem);
            NGA_Get(tA,loA.data(),hiA.data(),buffA.data(),ldA.data());
            // create local kCore tensors for permutation
            CoreTensorImpl* local_A_part = new CoreTensorImpl("Local A for permutation",dims_A_part);
            CoreTensorImpl* local_C_part = new CoreTensorImpl("data on Local C",dims_C_part);
            // copy data to local core tensor A
            local_A_part->data() = buffA;
            // do local permutations
            local_C_part->permute(local_A_part,Cinds,Ainds,1.0,0.0);

            // Get Access to local data on C
            double* C_data;
            int ld[ga_rank-1];
            NGA_Access(global_array_,lo,hi,&C_data,ld);
            // accumulate data to local C
            C_DAXPY(nelem,alpha,local_C_part->data().data(),1,C_data,1);
            // update data in GA
            NGA_Release_update(global_array_,lo,hi);

            delete local_A_part;
            delete local_C_part;
        }

//        scale(beta);

//        // Check the number of Cinds and Ainds
//        if(Cinds.size()!=ga_rank || Ainds.size()!= ga_rank)
//            throw std::runtime_error("The number of indices is inconsistent with Tensor rank!");

//        // Create a permutation map (all indices should be named differently)
//        std::vector<int> permutation_map(ga_rank);
//        for(int i=0; i<ga_rank; ++i) {
//            for (int j=0; j<ga_rank; ++j) {
//                if(Cinds[i].compare(Ainds[j]) == 0)
//                    permutation_map[i] = j;
//            }
//        }

//        // Check the dimensions after permutation
//        for(int i=0; i<ga_rank; ++i) {
//            if(ga_dims[i] != ga_dims[permutation_map[i]]) {
//                throw std::runtime_error("Permuted tensors do not have same dimensions!\n");
//            }
//        }

//        // Perform permutations
//        std::vector<int> lo(ga_rank);
//        int ld[1] = {1};
//        double buff[1];
//        iterate([&](const std::vector<size_t> &indices, double &value) {
//            // locate the position of target data (1 data at a time) slow?
//            for(int i=0; i<ga_rank; ++i) {
//                lo[permutation_map[i]] = indices[i];
//            }
//            // get target data
//            NGA_Get(tA,lo.data(),lo.data(),&buff,ld);
//            // add to the array
//            value += alpha*buff[0];
//        });

        GA_Sync();
    }
//    ambit::timer::timer_pop();

}

void GlobalArrayImpl::contract(
        ConstTensorImplPtr A,
        ConstTensorImplPtr B,
        const std::vector<std::string> &Cinds,
        const std::vector<std::string> &Ainds,
        const std::vector<std::string> &Binds,
        double alpha,
        double beta)
{   
    const GlobalArrayImpl* cA = dynamic_cast<const GlobalArrayImpl*>((A));
    int tA = cA->global_array_;
    const GlobalArrayImpl* cB = dynamic_cast<const GlobalArrayImpl*>((B));
    int tB = cB->global_array_;

//     A new way: copy needed data to local and use core tensor to do contraction
    scale(beta);

    // Find out what is on processor "me"
    int me = settings::rank;
    std::vector<int> loC(ga_rank,0);
    std::vector<int> hiC(ga_rank,0);
    if (ga_rank != 0) {
        NGA_Distribution(global_array_,me,loC.data(),hiC.data());
        // check if there is actually data on me
        if (loC[0] < 0) {
            GA_Sync();
            return;
        }
    }


    // find the place of Cinds in Ainds and Binds
    typedef std::pair<int,int> intPair;
    std::vector<intPair> C_ind_in_A;
    std::vector<intPair> C_ind_in_B;
    for (int iC=0; iC<Cinds.size(); ++iC) {
        for (int iA=0; iA<Ainds.size(); ++iA)
            if(Cinds[iC].compare(Ainds[iA]) == 0)
                C_ind_in_A.push_back(intPair(iC,iA));
        for (int iB=0; iB<Binds.size(); ++iB)
            if(Cinds[iC].compare(Binds[iB]) == 0)
                C_ind_in_B.push_back(intPair(iC,iB));
    }

    // The dimensions for local core tensors
    Dimension dims_A_part = A->dims();
    Dimension dims_B_part = B->dims();
    Dimension dims_C_part(Cinds.size());
    // Determine the local dims of C
    for (int iC=0; iC<Cinds.size(); ++iC) {
        dims_C_part[iC] = hiC[iC] - loC[iC] + 1;
    }
    // set original range of A
    std::vector<int> loA(Ainds.size());
    std::vector<int> hiA(Ainds.size());
    std::vector<int> ldA(Ainds.size());
    for(int iA=0; iA<Ainds.size(); ++iA) {
        loA[iA] = 0;
        hiA[iA] = A->dims()[iA] - 1;
        ldA[iA] = A->dims()[iA];
    }
    // change the range of A according to repeated indices in C
    for(int ind=0; ind<C_ind_in_A.size(); ++ind) {
        int indC = C_ind_in_A[ind].first;
        int indA = C_ind_in_A[ind].second;
        dims_A_part[indA] = dims_C_part[indC];
        loA[indA] = loC[indC];
        hiA[indA] = hiC[indC];
        ldA[indA] = dims_C_part[indC];
    }
    // set original range of B
    std::vector<int> loB(Binds.size());
    std::vector<int> hiB(Binds.size());
    std::vector<int> ldB(Binds.size());
    for(int iB=0; iB<Binds.size(); ++iB) {
        loB[iB] = 0;
        hiB[iB] = B->dims()[iB] - 1;
        ldB[iB] = B->dims()[iB];
    }
    // change the range of B according to repeated indices in C
    for(int ind=0; ind<C_ind_in_B.size(); ++ind) {
        int indC = C_ind_in_B[ind].first;
        int indB = C_ind_in_B[ind].second;
        dims_B_part[indB] = dims_C_part[indC];
        loB[indB] = loC[indC];
        hiB[indB] = hiC[indC];
        ldB[indB] = dims_C_part[indC];
    }
    // remove the first dimension in ldA and ldB
    ldA.erase(ldA.begin());
    ldB.erase(ldB.begin());

    ambit::timer::timer_push("Create Local Tensor");
    // create local kCore tensors for contraction
    CoreTensorImpl* local_A_part = new CoreTensorImpl("Local A for contraction",dims_A_part);
    CoreTensorImpl* local_B_part = new CoreTensorImpl("Local B for contraction",dims_B_part);
    CoreTensorImpl* local_C_part = new CoreTensorImpl("data on Local C",dims_C_part);
    ambit::timer::timer_pop();

    // Get the target data from A and B
    ambit::timer::timer_push("get data from A and B");
//        NGA_Get(tA,loA.data(),hiA.data(),buffA.data(),ldA.data());
    NGA_Get(tA,loA.data(),hiA.data(),local_A_part->data().data(),ldA.data());
//        NGA_Get(tB,loB.data(),hiB.data(),buffB.data(),ldB.data());
    NGA_Get(tB,loB.data(),hiB.data(),local_B_part->data().data(),ldB.data());
    ambit::timer::timer_pop();


    ambit::timer::timer_push("kCore contraction");

    // perform contraction locally
    local_C_part->contract(local_A_part,local_B_part,Cinds,Ainds,Binds,1.0,0.0);

    ambit::timer::timer_pop();
    if(ga_rank == 0) {
         single_value_ += alpha * local_C_part->data()[0];
    }
    else {
        ambit::timer::timer_push("update data");

        // Get Access to local data on C
        double *C_data;
        int ld[ga_rank-1];
        NGA_Access(global_array_,loC.data(),hiC.data(),&C_data,ld);

        // accumulate data to local C
        C_DAXPY(local_C_part->data().size(),alpha,local_C_part->data().data(),1,C_data,1);
        // update data in GA
        NGA_Release_update(global_array_,loC.data(),hiC.data());

        ambit::timer::timer_pop();
    }

     delete local_A_part;
     delete local_B_part;
     delete local_C_part;











    /// The second way: loop over each element in C and do the contraction
//    // find repeated indices in A and B and number of repeated elements
//    typedef std::pair<int,int> intPair;
//    std::vector<intPair> repeated_inds;
//    int contract_size = 1;
//    std::vector<int> ldA(Ainds.size(),1);
//    std::vector<int> ldB(Binds.size(),1);
//    for(int iA=0; iA<Ainds.size(); ++iA) {
//        for(int iB=0; iB<Binds.size(); ++iB) {
//            if(Ainds[iA].compare(Binds[iB])==0) {
//                repeated_inds.push_back(intPair(iA,iB));
//                contract_size *= cA->ga_dims[iA];
//                ldA[iA] = cA->ga_dims[iA];
//                ldB[iB] = cA->ga_dims[iA];
//            }
//        }
//    }
//    // Store the repeated indices in A and B and their dimensions
//    std::vector<std::string> Repeated_Ainds;
//    std::vector<std::string> Repeated_Binds;
//    bool repeated_same_order = true ; // this the repeated indices in A and B in the same order?
//    Dimension dims_A_part, dims_B_part;
//    for(int pair=0; pair<repeated_inds.size(); ++pair) {
//        int indA = repeated_inds[pair].first;
//        Repeated_Ainds.push_back(Ainds[indA]);
//        dims_A_part.push_back(A->dims()[indA]);
//        int indB = repeated_inds[pair].second;
//        Repeated_Binds.push_back(Binds[indB]);
//        dims_B_part.push_back(B->dims()[indB]);
//        if(pair>0 && indB < repeated_inds[pair-1].second) repeated_same_order = false;
//    }
//    // create local kCore tensors for contraction
//    CoreTensorImpl* Contract_A_part = new CoreTensorImpl("Local A for Contraction",dims_A_part);
//    CoreTensorImpl* Contract_B_part = new CoreTensorImpl("Local B for Contraction",dims_B_part);
//    CoreTensorImpl* Contract_Result = new CoreTensorImpl("Result Fake Tensor--Value" ,{});



//    // the first dimension should not be in ld
//    ldA.erase(ldA.begin());
//    ldB.erase(ldB.begin());

//    // assuming other indices are listed in C, create permutation map
//    std::vector<int> permutation_map(ga_rank);
//    for(int iC=0; iC<ga_rank; ++iC) {
//        for (int iA=0; iA<Ainds.size(); ++iA) {
//            if(Cinds[iC].compare(Ainds[iA]) == 0)
//                permutation_map[iC] = iA + 1; // use positive number to represent position in A, +1 to avoid 0
//        }
//        for (int iB=0; iB<Binds.size(); ++iB) {
//            if(Cinds[iC].compare(Binds[iB]) == 0)
//                permutation_map[iC] = -(iB + 1); // use negative number to represent position in B, +1 to avoid 0
//        }
//    }

//    // scale C
//    scale(beta);

//    // lo and hi range for A and B
//    std::vector<int> loA(Ainds.size());
//    std::vector<int> hiA(Ainds.size());
//    std::vector<int> loB(Binds.size());
//    std::vector<int> hiB(Binds.size());

//    // buff for reading data from A and B
//    std::vector<double> buffA(contract_size);
//    std::vector<double> buffB(contract_size);

//    // perform the contraction
//    iterate([&](const std::vector<size_t> &indices, double &value) {
//        // for non-repeating indices, find data location in A and B
//        for(int i=0; i<ga_rank; ++i) {
//            int target_ind = permutation_map[i];
//            if(target_ind > 0) {
//                loA[target_ind-1] = indices[i];
//                hiA[target_ind-1] = indices[i];
//            }
//            else if(target_ind < 0) {
//                loB[-target_ind-1] = indices[i];
//                hiB[-target_ind-1] = indices[i];
//            }
//        }
//        // for repeating indices, get data for corresponding dimensions
//        for(int ind=0; ind<repeated_inds.size(); ++ind) {
//            int repeated_A = repeated_inds[ind].first;
//            loA[repeated_A] = 0;
//            hiA[repeated_A] = cA->ga_dims[repeated_A] - 1;
//            int repeated_B = repeated_inds[ind].second;
//            loB[repeated_B] = 0;
//            hiB[repeated_B] = cB->ga_dims[repeated_B] - 1;
//        }

//        // get target data blocks
//        NGA_Get(tA,loA.data(),hiA.data(),buffA.data(),ldA.data());
//        NGA_Get(tB,loB.data(),hiB.data(),buffB.data(),ldB.data());
////        NGA_Get(tA,loA.data(),hiA.data(),Contract_A_part->data().data(),ldA.data());
////        NGA_Get(tB,loB.data(),hiB.data(),Contract_B_part->data().data(),ldB.data());
//        // sum up all data (contract). Note: call blas_ddot here should be much faster
//        if(repeated_same_order) {
//             value += alpha * C_DDOT(contract_size, buffA.data(), 1, buffB.data(), 1);
//        }
//        else {
//            Contract_A_part->data() = buffA;
//            Contract_B_part->data() = buffB;
//            Contract_Result->contract(Contract_A_part,Contract_B_part,{},Repeated_Ainds,Repeated_Binds,1.0,0.0);
//            value += alpha*Contract_Result->data()[0];
//        }
//        // put data into local core tensor and do the contraction
//    });

//    delete Contract_A_part;
//    delete Contract_B_part;
//    delete Contract_Result;

    GA_Sync();

}

std::map<std::string, TensorImplPtr> GlobalArrayImpl::syev(EigenvalueOrder order) const
{
    TensorImpl::squareCheck(this);
    // Allocate memory for diagonalization
    if(!MA_initialized()){
        int heap=3200000, stack=3200000;
        if(MA_init(C_DBL,stack,heap)==0)
            throw std::runtime_error("MA Initialization Failed");
    }


    GlobalArrayImpl *vectors = new GlobalArrayImpl("Eigenvectors of " + name(), dims());
    GlobalArrayImpl *vals = new GlobalArrayImpl("Eigenvalues of " + name(), {dims()[0]});
    // Save eigenvalues in local Core tensor
    // CoreTensorImpl *vals = new CoreTensorImpl("Eigenvalues of " + name(), {dims()[0]});


    int eigvector = vectors->global_array();
    // local copy of the eigenvalues
    std::vector<double> eigvalue(ga_dims[0]);
    // Call seq diagonalize
    GA_Diag_std_seq(global_array_,eigvector,eigvalue.data());
    // reverse order if needed
    if (order == kDescending)
        std::reverse(eigvalue.begin(),eigvalue.end());

    // update each processor's local data of the global array of eigenvalues
    int me = settings::rank;
    int lo[1],hi[1];
    NGA_Distribution(vals->global_array(),me,lo,hi);
    double* buff;
    int ld[1] = {1};
    NGA_Access(vals->global_array(),lo,hi,&buff,ld);
    std::copy(eigvalue.begin()+lo[0],eigvalue.begin()+hi[0]+1, buff);
    //update data in GA
    NGA_Release_update(vals->global_array(),lo,hi);

    //vals->data() = eigvalue;

    // gather rusults and return
    std::map<std::string, TensorImplPtr> results;
    results["eigenvalues"] = vals;
    results["eigenvectors"] = vectors;

    return results;
}

TensorImplPtr GlobalArrayImpl::power(double alpha, double condition) const
{
    TensorImpl::squareCheck(this);

    // get the eigenvalues and eigenvectors
    auto eigsys = syev(kAscending);
    const GlobalArrayImpl* eigvtr = dynamic_cast<const GlobalArrayImpl*>(eigsys["eigenvectors"]);
    GlobalArrayImpl* eigval = dynamic_cast<GlobalArrayImpl*>(eigsys["eigenvalues"]);

    // create a matrix with power(eigenvalues,alpha) at diagonal
    GlobalArrayImpl *results = new GlobalArrayImpl(name()+"^"+std::to_string(alpha), dims());
    results->set(0.0);

    double max = std::get<0>(eigval->max());
    eigval->iterate([&](const std::vector<size_t> &indices, double &value) {
        if (alpha < 0 && fabs(value) < condition*max)
            value = 0.0;
        else {
            value = pow(value,alpha);
            if (std::isfinite(value) == false) value = 0.0;
        }
    });

    GA_Set_diagonal(results->global_array(),eigval->global_array());

    // results = L^-1 . R^alpha . L
    int results_ga = results->global_array();
    GlobalArrayImpl *tmp = new GlobalArrayImpl("tmp", dims());
    int tmp_ga = tmp->global_array();
    int length = (int)dims()[0];
    GA_Dgemm('N','N',length,length,length,1.0,results_ga,eigvtr->global_array(),0.0,tmp_ga);
    GA_Dgemm('T','N',length,length,length,1.0,eigvtr->global_array(),tmp_ga,0.0,results_ga);

    // Need to manually delete the tensors in the diag map
    for (auto& el : eigsys) delete el.second;
    delete tmp;

    return results;
}

void GlobalArrayImpl::iterate(const std::function<void(const std::vector<size_t> &, double &)> &func)
{
    if (ga_rank == 0) {
        std::vector<size_t> indices;
        func(indices,single_value_);
        return;
    }

    // Find out what is on processor "me"
    int me = settings::rank;
    int lo[ga_rank],hi[ga_rank];
    NGA_Distribution(global_array_,me,lo,hi);

    // Get data to local copy "buff", with leading dimention "ld"
    double* buff;
    int ld[ga_rank-1];
    NGA_Access(global_array_,lo,hi,&buff,ld);

    // indices and addressing array
    std::vector<size_t> indices(ga_rank, 0);
    std::vector<size_t> addressing(ga_rank, 1);

    // form addressing array
    for (int n=ga_rank-2; n >= 0; --n) {
        addressing[n] = addressing[n+1] * ld[n];
    }

    // find out total number of elements
    size_t nelem = (size_t) (hi[0]-lo[0]+1)*addressing[0];

    //loop over local data and perform function
    for (size_t n=0; n < nelem; ++n) {
        size_t d=n;
        for(size_t k=0; k<(size_t)ga_rank; ++k) {
            indices[k] = d / addressing[k] + lo[k];
            d = d % addressing[k];
        }
        func(indices, buff[n]);
    }

    //update data in GA
    NGA_Release_update(global_array_,lo,hi);

}

void GlobalArrayImpl::citerate(const std::function<void(const std::vector<size_t> &, const double &)> &func) const
{
    if (ga_rank == 0) {
        std::vector<size_t> indices;
        func(indices,single_value_);
        return;
    }

    // Find out what is on processor "me"
    int me = settings::rank;
    int lo[ga_rank],hi[ga_rank];
    NGA_Distribution(global_array_,me,lo,hi);

    // Get data to local copy "buff", with leading dimention "ld"
    double* buff;
    int ld[ga_rank-1];
    NGA_Access(global_array_,lo,hi,&buff,ld);

    // indices and addressing array
    std::vector<size_t> indices(ga_rank, 0);
    std::vector<size_t> addressing(ga_rank, 1);

    // form addressing array
    for (int n=ga_rank-2; n >= 0; --n) {
        addressing[n] = addressing[n+1] * ld[n];
    }

    // find out total number of elements
    size_t nelem = (size_t) (hi[0]-lo[0]+1)*addressing[0];

    //loop over local data and perform function
    for (size_t n=0; n < nelem; ++n) {
        size_t d=n;
        for(size_t k=0; k<(size_t)ga_rank; ++k) {
            indices[k] = d / addressing[k] + lo[k];
            d = d % addressing[k];
        }
        func(indices, buff[n]);
    }

}


}
}
