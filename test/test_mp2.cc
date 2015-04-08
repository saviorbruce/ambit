#include <string>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <assert.h>

#include <ambit/tensor.h>
#include <ambit/io/io.h>
#include <ambit/helpers/psi4/io.h>

using namespace ambit;

TensorType tensor_type = kCore;

Tensor build(const std::string& name, const Dimension& dims)
{
    return Tensor::build(tensor_type, name, dims);
}

Tensor build_and_load(const std::string& file35, const std::string& toc, const Dimension& AO)
{
    Tensor X = build(toc, AO);
    helpers::psi4::load_matrix(file35, toc, X);
    return X;
}

Tensor load_overlap(const std::string& file35, const Dimension& AO)
{
    return build_and_load(file35, "SO-basis Overlap Ints", AO);
}

Tensor load_1e_hamiltonian(const std::string& file35, const Dimension& AO)
{
    Tensor H = build("H", AO);

    Tensor T = build_and_load(file35, "SO-basis Kinetic Energy Ints", AO);
    Tensor V = build_and_load(file35, "SO-basis Potential Energy Ints", AO);

    H("p,q") = T("p,q") + V("p,q");

    return H;
}

Tensor load_2e(const Dimension& AO)
{
    Tensor g = build("g", AO);
    helpers::psi4::load_iwl("test.33", g);
    return g;
}

void mp2()
{
    int nirrep, nso;
    double Enuc = 0.0, Eref = 0.0;

    {
        ambit::io::File file32("test.32", ambit::io::kOpenModeOpenExisting);

        file32.read("::Num. irreps", &nirrep, 1);
        printf("nirrep = %d\n", nirrep);
        assert(nirrep == 1);

        file32.read("::Num. SO", &nso, 1);
        printf("nso = %d\n", nso);

        file32.read("::Nuclear rep. energy", &Enuc, 1);
        file32.read("::SCF energy", &Eref, 1);
    }

    // Define dimension objects
    Dimension AO2 = {(size_t)nso, (size_t)nso};
    Dimension AO4 = {(size_t)nso, (size_t)nso, (size_t)nso, (size_t)nso};

    // Build tensors
    Tensor S = load_overlap("test.35", AO2);
    printf("norm of S is %lf\n", S.norm());
    Tensor H = load_1e_hamiltonian("test.35", AO2);
    Tensor g = load_2e(AO4);
    printf("norm of g is %lf\n", g.norm());

    Tensor Ft = build("Ft", AO2);
    Tensor Smhalf = S.power(-0.5);
//    Smhalf.print(stdout, true);

    Ft("i,j") = Smhalf("mu,i") * Smhalf("nu,j") * H("mu,nu");
//    Ft.print(stdout, true);
    auto Feigen = Ft.syev(kAscending);
//    Feigen["eigenvectors"].print(stdout, true);

    Tensor C = build("C", AO2);
    C("i,j") = Smhalf("k,j") * Feigen["eigenvectors"]("i,k");
//    C.print(stdout, true);

    size_t ndocc = 5;
    Tensor Cdocc = build("C", {ndocc, (size_t)nso});
    IndexRange CtoCdocc = { {0,ndocc}, {0,(size_t)nso}};
    //Cdocc.slice(C, CtoCdocc, CtoCdocc);
    Cdocc(CtoCdocc) = C(CtoCdocc);

    // Form initial D
    Tensor D = build("D", AO2);
    D("mu,nu") = Cdocc("i,mu") * Cdocc("i,nu");

    Tensor F = build("F", AO2);

    // start SCF iteration
    bool converged = false;
    double Eelec = 0.0, Eold = 0.0;
    int iter = 1, maxiter = 15;
    do {
        F("mu,nu") = H("mu,nu");
        F("mu,nu") += D("rho,sigma") * (2.0 * g("mu,nu,rho,sigma") - g("mu,rho,nu,sigma"));
//        F.print(stdout, true);

        // Calculate energy
        Eelec = D("mu,nu") * (H("mu,nu") + F("mu,nu"));

        if (settings::rank == 0)
            printf("  @RHF iter %5d: dE = %20.14lf\n", iter++, Eelec - Eold);

        // Transform the Fock matrix
        Ft("i,j") = Smhalf("mu,i") * Smhalf("nu,j") * F("mu,nu");

        // Diagonalize Fock matrix
        Feigen = Ft.syev(kAscending);

        // Construct new SCF eigenvector matrix.
        C("i,j") = Smhalf("k,j") * Feigen["eigenvectors"]("i,k");

        // Form new density matrix
        Cdocc(CtoCdocc) = C(CtoCdocc);
        D("mu,nu") = Cdocc("i,mu") * Cdocc("i,nu");
//        D.print(stdout, true);

        if (std::fabs(Eelec - Eold) < 1.0e-10) {
            converged = true;
            printf("  HF has converged!\n  Final HF energy:        %20.14lf\n", Enuc + Eelec );
        }
        Eold = Eelec;

        if (iter > maxiter) {
            printf("  HF has not converged in %d iterations!\n", maxiter);
            break;
        }
    } while (!converged);

//    MP2

    double e_scf = Enuc + Eelec;
    size_t nall = (size_t)nso;
    //prepare Cv matrix to use together with Co
    size_t nvocc = nall - ndocc;
    Tensor Cvocc = build("Cv", {nvocc, nall});
    Cvocc({{0,nvocc},{0,nall}}) = C({{ndocc,nall},{0,nall}});

    // AO to MO transformation
    Tensor Gpqrb = build("Gpqrb",{nall,nall,nall,nvocc});
    Tensor Gpqjb = build("Gpqrb",{nall,nall,ndocc,nvocc});
    Tensor Gpajb = build("Gpqrb",{nall,nvocc,ndocc,nvocc});
    Tensor Giajb = build("Giajb",{ndocc,nvocc,ndocc,nvocc});
    Gpqrb("p,q,r,b") = g("p,q,r,s")*Cvocc("b,s");
    Gpqjb("p,q,j,b") = Gpqrb("p,q,r,b")*Cdocc("j,r");
    Gpajb("p,a,j,b") = Gpqjb("p,q,j,b")*Cvocc("a,q");
    Giajb("i,a,j,b") = Gpajb("p,a,j,b")*Cdocc("i,p");

    // the energy eigenvalues

    Tensor t_eigev = Tensor::build(kCore, "eigenvalues", {nall});
    t_eigev() = Feigen["eigenvalues"]();
    std::vector<double> e_eigev = t_eigev.data();

    // Construct denominators

    Tensor Dijab = build("Dijab",{ndocc,ndocc,nvocc,nvocc});
    Dijab.iterate([&](const std::vector<size_t>& indices, double& value) {
        value = 1.0/(e_eigev[indices[0]]+e_eigev[indices[1]]-e_eigev[indices[2]+ndocc]-e_eigev[indices[3]+ndocc]);
    });

//    for(size_t i=0;i<ndocc;++i) {
//        for(size_t j=0;j<ndocc;++j) {
//            for(size_t a=0;a<nvocc;++a) {
//                for(size_t b=0;b<nvocc;++b) {
//                    size_t ijab = i*ndocc*nvocc*nvocc+j*nvocc*nvocc+a*nvocc+b;
//                    Dijab.data()[ijab]=1.0/(e_eigev[i]+e_eigev[j]-e_eigev[a+ndocc]-e_eigev[b+ndocc]);
//                }
//            }
//        }
//    }




    // Compute MP2 energy
    Tensor Aijab = build("Aijab",{ndocc,ndocc,nvocc,nvocc});
    Aijab("i,j,a,b") = Giajb("i,a,j,b") - Giajb("i,b,j,a");
    Tensor Aijab2 = build("Aijab2",{ndocc,ndocc,nvocc,nvocc});
    Aijab2("i,j,a,b") = Aijab("i,j,a,b")*Aijab("i,j,a,b");
    Tensor Gijab2 = build("Gijab2",{ndocc,ndocc,nvocc,nvocc});
    Gijab2("i,j,a,b") = Giajb("i,a,j,b")*Giajb("i,a,j,b");

    double e_aa = Aijab2("i,j,a,b") * Dijab("i,j,a,b")/4.0;
    double e_bb = e_aa;
    double e_ab = Gijab2("i,j,a,b") * Dijab("i,j,a,b");
    double e_mp2 = e_aa + e_bb + e_ab;

    printf("  MP2 Correlation Energy: %20.14lf\n", e_mp2);
    printf("  Total MP2 Energy:       %20.14lf\n", e_mp2+e_scf);


//    C.print(stdout, true);
    C.iterate([](const std::vector<size_t>& /*indices*/, double& value) {
        value += 1.0;
    });
//    C.print(stdout, true);

//    C.citerate([](const std::vector<size_t>& indices, const double& value) {
//        printf("rank %d: C[%lu, %lu] %lf\n", settings::rank, indices[0], indices[1], value);
//    });

//    g.citerate([](const std::vector<size_t>& indices, const double& value) {
//        printf("g[%lu, %lu, %lu, %lu] %lf\n", indices[0], indices[1], indices[2], indices[3], value);
//    });
}

int main(int argc, char* argv[])
{
    srand(time(nullptr));
    ambit::initialize(argc, argv);

    if (argc > 1) {
        if (settings::distributed_capable && strcmp(argv[1], "cyclops") == 0) {
            tensor_type = kDistributed;
            printf("  *** Testing distributed tensors. ***\n");
        }
        else {
            printf("  *** Unknown parameter given ***\n");
            printf("  *** Testing core tensors.   ***\n");
        }
    }
    mp2();

    ambit::finalize();
    return EXIT_SUCCESS;
}

