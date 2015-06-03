#include <string>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <assert.h>

#include <ambit/print.h>
#include <ambit/tensor.h>
#include <ambit/io/io.h>
#include <ambit/helpers/psi4/io.h>
#include <ambit/timer.h>

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

void hf()
{
    int nirrep, nso;
    double Enuc = 0.0, Eref = 0.0;

    {
        ambit::io::File file32("test.32", ambit::io::kOpenModeOpenExisting);

        file32.read("::Num. irreps", &nirrep, 1);
        print("nirrep = %d\n", nirrep);
        assert(nirrep == 1);

        file32.read("::Num. SO", &nso, 1);
        print("nso = %d\n", nso);

        file32.read("::Nuclear rep. energy", &Enuc, 1);
        file32.read("::SCF energy", &Eref, 1);
    }

    // Define dimension objects
    Dimension AO2 = {(size_t)nso, (size_t)nso};
    Dimension AO4 = {(size_t)nso, (size_t)nso, (size_t)nso, (size_t)nso};

    // Build tensors
    Tensor S = load_overlap("test.35", AO2);
    print("norm of S is %lf\n", S.norm());
    Tensor H = load_1e_hamiltonian("test.35", AO2);
    Tensor g = load_2e(AO4);
    print("norm of g is %lf\n", g.norm());

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

    Tensor Cdocc = build("C", {5, (size_t)nso});

    size_t ndocc = 5;
    IndexRange CtoCdocc = { {0,ndocc}, {0,(size_t)nso}};
    //Cdocc.slice(C, CtoCdocc, CtoCdocc);
    Cdocc(CtoCdocc) = C(CtoCdocc);

    // Form initial D
    Tensor D = build("D", AO2);
    D("mu,nu") = Cdocc("i,mu") * Cdocc("i,nu");
//    D.print();

    Tensor F = build("F", AO2);

//    // Build 2<pq|rs>-<pr|qs>
//    Tensor g2_g = build("2<pq|rs>-<pr|qs>", AO4);
//    g2_g("mu,nu,rho,sigma") = 2.0 * g("mu,nu,rho,sigma") - g("mu,rho,nu,sigma");


    // start SCF iteration
    bool converged = false;
    double Eelec = 0.0, Eold = 0.0;
    int iter = 1, maxiter = 100;
    do {
        ambit::timer::timer_push("HF iteration");

        F("mu,nu") = H("mu,nu");
        F("mu,nu") += D("rho,sigma") * (2.0 * g("mu,nu,rho,sigma") - g("mu,rho,nu,sigma"));
//        F("mu,nu") += D("rho,sigma") * g2_g("mu,nu,rho,sigma");

        // Calculate energy
        Eelec = D("mu,nu") * (H("mu,nu") + F("mu,nu"));

        print("  @RHF iter %5d: %20.14lf\n", iter, Enuc + Eelec);

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

        if (std::fabs(Eelec - Eold) < 1.0e-13) converged = true;
        Eold = Eelec;

        ambit::timer::timer_pop();

        if (iter++ > maxiter)
            break;
    } while (!converged);

}

int main(int argc, char* argv[])
{
    srand(time(nullptr));
    ambit::settings::timers = true;
    ambit::initialize(argc, argv);

    if (argc > 1) {
        printf("settings::distributed_capable: %d\n", settings::distributed_capable);
        if (settings::distributed_capable && strcmp(argv[1], "cyclops") == 0) {
            tensor_type = kDistributed;
            ambit::print("  *** Testing distributed tensors. ***\n");
            ambit::print("      Running in %d processes.\n", ambit::settings::nprocess);
        }
        else if (settings::distributed_capable && strcmp(argv[1], "ga") == 0) {
            tensor_type = kGlobalArray;
            ambit::print("  *** Testing distributed tensors. ***\n");
            ambit::print("      Running in %d processes.\n", ambit::settings::nprocess);
        }
        else {
            ambit::print("  *** Unknown parameter given ***\n");
            ambit::print("  *** Testing core tensors.   ***\n");
        }
    }



    hf();

    ambit::finalize();
    return EXIT_SUCCESS;
}
