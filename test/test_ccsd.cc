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

void ccsd()
{
    int nirrep, nso , n_docc;
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

        file32.read("::Closed shells per irrep", &n_docc, 1);
        print("ndocc = %d\n", n_docc);
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

    // Build 2<pq|rs>-<pr|qs>
    Tensor g2_g = build("2<pq|rs>-<pr|qs>", AO4);
    g2_g("mu,nu,rho,sigma") = 2.0 * g("mu,nu,rho,sigma") - g("mu,rho,nu,sigma");

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

    size_t ndocc = (size_t)n_docc;
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
    int iter = 1, maxiter = 100;

    do {
        ambit::timer::timer_push("HF iteration");


        F("mu,nu") = H("mu,nu");
        F("mu,nu") += D("rho,sigma") * ( 2.0*g("mu,nu,rho,sigma") - g("mu,rho,nu,sigma"));  // Original


//        F("mu,nu") += D("rho,sigma") * g2_g("mu,nu,rho,sigma");
//        F.print(stdout, true);

        // Calculate energy
        Eelec = D("mu,nu") * (H("mu,nu") + F("mu,nu"));



        // Transform the Fock matrix
        Ft("i,j") = Smhalf("mu,i") * Smhalf("nu,j") * F("mu,nu");

        // Diagonalize Fock matrix
        Feigen = Ft.syev(kAscending);

        // Construct new SCF eigenvector matrix.
        C("i,j") = Smhalf("k,j") * Feigen["eigenvectors"]("i,k");

        // Form new density matrix
        Cdocc(CtoCdocc) = C(CtoCdocc);
        Tensor D_new = build("D new", AO2);
        D_new("mu,nu") = Cdocc("i,mu") * Cdocc("i,nu");

        // Compute RMS of D matrix
        Tensor delta_D = build("Difference in D matrix", AO2);
        delta_D("mu,nu") = D_new("mu,nu") - D("mu,nu");
        double D_change = delta_D("mu,nu") * delta_D("mu,nu");

        print("  @RHF iter %5d: Escf = %17.12f  dE = %12.5e  RMS(D) = %12.5e\n", iter++, Enuc + Eelec,Eelec-Eold,std::sqrt(D_change));

        if (std::fabs(Eelec - Eold) < 1.0e-13 && std::sqrt(D_change) < 1.0e-10) {
            converged = true;
            print("  HF has converged!\n  Final HF energy:        %20.14lf\n", Enuc + Eelec );
        }
        else {
            Eold = Eelec;
            D = D_new;
        }

        ambit::timer::timer_pop();

        if (iter > maxiter) {
            print("  HF has not converged in %d iterations!\n", maxiter);
            break;
        }
    } while (!converged);

//    MP2

    double e_scf = Enuc + Eelec;
    size_t nall = (size_t)nso;
    //prepare Cv matrix to use together with Co
    size_t nvocc = nall - ndocc;

    ambit::timer::timer_push("Integral Transformation");
    Tensor Cvocc = build("Cv", {nvocc, nall});
    Cvocc({{0,nvocc},{0,nall}}) = C({{ndocc,nall},{0,nall}});

    // Full AO to MO transformation
    Tensor Guvxs = build("intermediate Guvxs",{nall,nall,nall,nall});
    Tensor Guvrs = build("intermediate Guvrs",{nall,nall,nall,nall});
    Tensor Guqrs = build("intermediate Guqrs",{nall,nall,nall,nall});
    Tensor Gpqrs = build("MO space TEI: <pq|rs>",{nall,nall,nall,nall});
    Guvxs("u,v,x,s") = g("u,v,x,y")*C("s,y");
    Guvrs("u,v,r,s") = Guvxs("u,v,x,s")*C("r,x");
    Guqrs("u,q,r,s") = Guvrs("u,v,r,s")*C("q,v");
    Gpqrs("p,r,q,s") = Guqrs("u,q,r,s")*C("p,u");   // <pr|qs> = (pq|rs)

    ambit::timer::timer_pop();

    /* Slice the G tensor into 6 types of TEIs:
     * <ia|bc>, <ij|ab>, <ij|ka>, <ab|cd>, <ia|jb>, <ij|kl> */

    ambit::timer::timer_push("Slice into 6 types of TEIs");

    Tensor Giabc = build("Giabc",{ndocc,nvocc,nvocc,nvocc});
    Giabc({{0,ndocc},{0,nvocc},{0,nvocc},{0,nvocc}})=Gpqrs({{0,ndocc},{ndocc,nall},{ndocc,nall},{ndocc,nall}});
    Tensor Gijab = build("Gijab",{ndocc,ndocc,nvocc,nvocc});
    Gijab({{0,ndocc},{0,ndocc},{0,nvocc},{0,nvocc}})=Gpqrs({{0,ndocc},{0,ndocc},{ndocc,nall},{ndocc,nall}});
    Tensor Gijka = build("Gijka",{ndocc,ndocc,ndocc,nvocc});
    Gijka({{0,ndocc},{0,ndocc},{0,ndocc},{0,nvocc}})=Gpqrs({{0,ndocc},{0,ndocc},{0,ndocc},{ndocc,nall}});
    Tensor Gabcd = build("Gabcd",{nvocc,nvocc,nvocc,nvocc});
    Gabcd({{0,nvocc},{0,nvocc},{0,nvocc},{0,nvocc}})=Gpqrs({{ndocc,nall},{ndocc,nall},{ndocc,nall},{ndocc,nall}});
    Tensor Giajb = build("Giajb",{ndocc,nvocc,ndocc,nvocc});
    Giajb({{0,ndocc},{0,nvocc},{0,ndocc},{0,nvocc}})=Gpqrs({{0,ndocc},{ndocc,nall},{0,ndocc},{ndocc,nall}});
    Tensor Gijkl = build("Gijkl",{ndocc,ndocc,ndocc,ndocc});
    Gijkl({{0,ndocc},{0,ndocc},{0,ndocc},{0,ndocc}})=Gpqrs({{0,ndocc},{0,ndocc},{0,ndocc},{0,ndocc}});


    /* <ab|ci> is more efficient when contracting*/
    Tensor Gabci = build("Gabci",{nvocc,nvocc,nvocc,ndocc});
    Gabci({{0,nvocc},{0,nvocc},{0,nvocc},{0,ndocc}})=Gpqrs({{ndocc,nall},{ndocc,nall},{ndocc,nall},{0,ndocc}});
    /* <ai|bc> is more efficient when contracting*/
    Tensor Gaibc = build("Gaibc",{nvocc,ndocc,nvocc,nvocc});
    Gaibc({{0,nvocc},{0,ndocc},{0,nvocc},{0,nvocc}})=Gpqrs({{ndocc,nall},{0,ndocc},{ndocc,nall},{ndocc,nall}});


    ambit::timer::timer_pop();

    // the energy eigenvalues

    Tensor t_eigev = Tensor::build(kCore, "eigenvalues", {nall});
    IndexRange all = {{0L, nall}};
    t_eigev(all) = Feigen["eigenvalues"](all);
//    std::vector<double> e_eigev = t_eigev.data();


    // Construct denominators
    // We only need D_IA and D_IjAb in Spin-adapted version

    ambit::timer::timer_push("Constructing denominators");

    Tensor Dia = build("Dia",{ndocc,nvocc});
    Dia.iterate([&](const std::vector<size_t>& indices, double& value) {
//        value = 1.0/(e_eigev[indices[0]]-e_eigev[indices[1]+ndocc]);
        value = 1.0/(t_eigev.data()[indices[0]]-t_eigev.data()[indices[1]+ndocc]);
    });

    Tensor Dijab = build("Dijab",{ndocc,ndocc,nvocc,nvocc});
    Dijab.iterate([&](const std::vector<size_t>& indices, double& value) {
        value = 1.0/(t_eigev.data()[indices[0]]+t_eigev.data()[indices[1]]
                -t_eigev.data()[indices[2]+ndocc]-t_eigev.data()[indices[3]+ndocc]);
    });

    ambit::timer::timer_pop();

    /* Close-shell CCSD */



    //Prepare the MO Basis Fock Matrix
//    Tensor Fmo = build("MO basis Fock matrix",{nall,nall});
//    Fmo("p,q") = F("mu,nu") * C("p,mu") * C("q,nu");
//    Fmo.print(stdout, true);

    // Build the Initial-Guess Cluster Amplitudes
    //spin-adapted version: we only need the TIA and TIjAb amplitudes
    //the initial TIJAB = TIjAb-TJiAb;
    Tensor T1 = build("T1 amplitude",{ndocc,nvocc});
    Tensor T2 = build("T2 amplitude",{ndocc,ndocc,nvocc,nvocc});
    T2("i,j,a,b") = Gijab("i,j,a,b")*Dijab("i,j,a,b");

    // Test the MP2 energy
    double e_mp2 = (2*T2("i,j,a,b")-T2("j,i,a,b"))*Gijab("i,j,a,b");
    print("  MP2 Correlation Energy: %20.14lf\n", e_mp2);
    print("  Total MP2 Energy:       %20.14lf\n\n", e_scf+e_mp2);

    // Start CC iteration
    double e_ccsd = 0.0;
    converged = false;
    iter = 1, maxiter = 50;
    do {

        ambit::timer::timer_push("CCSD iteration");

        //Tau_tilde, Tau, and Tau_tt
        Tensor Tau_t = build("Tau tilde",{ndocc,ndocc,nvocc,nvocc});
        Tau_t("i,j,a,b") = T2("i,j,a,b");
        Tau_t("i,j,a,b") += 0.5*T1("i,a")*T1("j,b");
        Tensor Tau = build("Tau",{ndocc,ndocc,nvocc,nvocc});
        Tau("i,j,a,b") = T2("i,j,a,b");
        Tau("i,j,a,b") += T1("i,a")*T1("j,b");
        Tensor Tau_tt = build("Tau_tt",{ndocc,ndocc,nvocc,nvocc});
        Tau_tt("i,j,a,b") = T2("i,j,a,b");
        Tau_tt("i,j,a,b") += 2.0*T1("i,a")*T1("j,b");



        //Form the intermediates
        // Fae
        Tensor Fae = build("Fae",{nvocc,nvocc});
        Fae("a,e") = T1("m,f")*(2*Giabc("m,a,f,e")-Giabc("m,a,e,f"));
        Fae("a,e") -= Tau_t("m,n,a,f")*(2*Gijab("m,n,e,f")-Gijab("n,m,e,f"));

        // Fmi
        Tensor Fmi = build("Fmi",{ndocc,ndocc});
        Fmi("m,i") = T1("n,e")*(2*Gijka("m,n,i,e")-Gijka("n,m,i,e"));
        Fmi("m,i") += Tau_t("i,n,e,f")*(2*Gijab("m,n,e,f")-Gijab("n,m,e,f"));

        // Fme
        Tensor Fme = build("Fme",{ndocc,nvocc});
        Fme("m,e") = T1("n,f")*(2*Gijab("m,n,e,f")-Gijab("n,m,e,f"));

        // Wmnij
        Tensor Wmnij = build("Wmnij",{ndocc,ndocc,ndocc,ndocc});
        Wmnij("m,n,i,j") = Gijkl("m,n,i,j");
        Wmnij("m,n,i,j") += T1("j,e")*Gijka("m,n,i,e");
        Wmnij("m,n,i,j") += T1("i,e")*Gijka("n,m,j,e");
        Wmnij("m,n,i,j") += 0.5*Tau("i,j,e,f")*Gijab("m,n,e,f");

        // Wabef
        Tensor Wabef = build("Wabef",{nvocc,nvocc,nvocc,nvocc});
        Wabef("a,b,e,f") = Gabcd("a,b,e,f");
        // Original
        Wabef("a,b,e,f") -= T1("m,b")*Giabc("m,a,f,e"); // This is the most expensive step
        // improved
//        Wabef("a,b,e,f") -= T1("m,b")*Gaibc("a,m,e,f"); // This is still 10 times slower than the T1("m,a")*Giabc("m,b,e,f") step

        // try use permutations
//        Tensor T1_tr = build("T1 trans",{nvocc,ndocc});
//        T1_tr("a,i") = T1("i,a");
//        Tensor Wtmp = build("W_tmp",{nvocc,nvocc,nvocc,nvocc});
//        Wtmp("a,f,e,b") -= T1_tr("b,m")*Gabci("a,f,e,m");
//        Wabef("a,b,e,f") += Wtmp("a,f,e,b");
//                {
//                    Tensor Wefab = build("Wefab", {nvocc, nvocc, nvocc, nvocc});
//                    Wefab("e,f,a,b") = T1("m,b") * Gabci("e,f,a,m");
//                    Wabef("a,b,e,f") -= Wefab("e,f,a,b");
//                }

        Wabef("a,b,e,f") -= T1("m,a")*Giabc("m,b,e,f");
        Wabef("a,b,e,f") += 0.5*Tau("m,n,a,b")*Gijab("m,n,e,f");

        // Wmbej abab case
        Tensor Wmbej = build("Wmbej",{ndocc,nvocc,nvocc,ndocc});
        Wmbej("m,b,e,j") = Gijab("m,j,e,b");
        Wmbej("m,b,e,j") += T1("j,f")*Giabc("m,b,e,f");
        Wmbej("m,b,e,j") -= T1("n,b")*Gijka("n,m,j,e");
        Wmbej("m,b,e,j") -= 0.5*Tau_tt("j,n,f,b")*Gijab("m,n,e,f");
        Wmbej("m,b,e,j") += 0.5*T2("n,j,f,b")*(2*Gijab("m,n,e,f")-Gijab("n,m,e,f"));

        //Wmbej abba case
        Tensor W_MbeJ= build("WMbeJ",{ndocc,nvocc,nvocc,ndocc});
        W_MbeJ("m,b,e,j") = -Giajb("m,b,j,e");
        W_MbeJ("m,b,e,j") -= T1("j,f")*Giabc("m,b,f,e");
        W_MbeJ("m,b,e,j") += T1("n,b")*Gijka("m,n,j,e");
        W_MbeJ("m,b,e,j") += 0.5*Tau_tt("j,n,f,b")*Gijab("n,m,e,f");

        //Compute new T1 and T2 amplitudes
        //new T1
        Tensor t1n = build("T1 new",{ndocc,nvocc});
        t1n("i,a") = T1("i,e")*Fae("a,e");
        t1n("i,a") -= T1("m,a")*Fmi("m,i");
        t1n("i,a") += Fme("m,e")*(2*T2("i,m,a,e")-T2("m,i,a,e"));
        t1n("i,a") += T1("m,e")*(2*Gijab("i,m,a,e")-Giajb("m,a,i,e"));
        t1n("i,a") -= T2("m,n,a,e")*(2*Gijka("m,n,i,e")-Gijka("n,m,i,e"));
        t1n("i,a") += T2("i,m,e,f")*(2*Giabc("m,a,f,e")-Giabc("m,a,e,f"));
        Tensor T1n = build("copy of t1n",{ndocc,nvocc});
        T1n("i,a") = t1n("i,a");
        t1n("i,a") =  T1n("i,a")*Dia("i,a");

        //new T2
        Tensor t2n = build("T2 new",{ndocc,ndocc,nvocc,nvocc});
        t2n("i,j,a,b") = Gijab("i,j,a,b");
        Fae("b,e") -= 0.5*T1("m,b")*Fme("m,e");
        t2n("i,j,a,b") += T2("i,j,a,e")*Fae("b,e");
        t2n("i,j,a,b") += T2("i,j,e,b")*Fae("a,e");
        Fmi("m,j") += 0.5*T1("j,e")*Fme("m,e");
        t2n("i,j,a,b") -= T2("i,m,a,b")*Fmi("m,j");
        t2n("i,j,a,b") -= T2("m,j,a,b")*Fmi("m,i");
        t2n("i,j,a,b") += Tau("m,n,a,b")*Wmnij("m,n,i,j");
        t2n("i,j,a,b") += Tau("i,j,e,f")*Wabef("a,b,e,f");
        t2n("i,j,a,b") += (T2("i,m,a,e")-T2("m,i,a,e"))*Wmbej("m,b,e,j");
        t2n("i,j,a,b") -= T1("i,e")*T1("m,a")*Gijab("m,j,e,b");
        t2n("i,j,a,b") += T2("i,m,a,e")*(Wmbej("m,b,e,j")+W_MbeJ("m,b,e,j"));
        t2n("i,j,a,b") += T2("m,i,b,e")*W_MbeJ("m,a,e,j");
        t2n("i,j,a,b") -= T1("i,e")*T1("m,b")*Giajb("m,a,j,e");
        t2n("i,j,a,b") += T2("m,j,a,e")*W_MbeJ("m,b,e,i");
        t2n("i,j,a,b") -= T1("j,e")*T1("m,a")*Giajb("m,b,i,e");
        t2n("i,j,a,b") += (T2("j,m,b,e")-T2("m,j,b,e"))*Wmbej("m,a,e,i");
        t2n("i,j,a,b") -= T1("j,e")*T1("m,b")*Gijab("m,i,e,a");
        t2n("i,j,a,b") += T2("j,m,b,e")*(Wmbej("m,a,e,i")+W_MbeJ("m,a,e,i"));
        t2n("i,j,a,b") += T1("i,e")*Giabc("j,e,b,a");
        t2n("i,j,a,b") += T1("j,e")*Giabc("i,e,a,b");
        t2n("i,j,a,b") -= T1("m,a")*Gijka("i,j,m,b");
        t2n("i,j,a,b") -= T1("m,b")*Gijka("j,i,m,a");
        Tensor T2n = build("copy of t2n",{ndocc,ndocc,nvocc,nvocc});
        T2n("i,j,a,b") = t2n("i,j,a,b");
        t2n("i,j,a,b") = T2n("i,j,a,b") * Dijab("i,j,a,b");
//        t2n.print(stdout,true);

        //compute new CCSD energy
        Tensor Tau_new = build("new Tau",{ndocc,ndocc,nvocc,nvocc});
        Tau_new("i,j,a,b") = t2n("i,j,a,b");
        Tau_new("i,j,a,b") += t1n("i,a")*t1n("j,b");
        double Eccn =  Gijab("i,j,a,b")*(2*Tau_new("i,j,a,b")-Tau_new("j,i,a,b"));
        //compute RMS in T1 and T2 amplitudes
        Tensor delta_t1 = build("difference in T1",{ndocc,nvocc});
        delta_t1("i,a") = t1n("i,a")-T1("i,a");
        double T1_change = delta_t1("i,a")*delta_t1("i,a");
        Tensor delta_t2 = build("difference in T2",{ndocc,ndocc,nvocc,nvocc});
        delta_t2("i,j,a,b") = t2n("i,j,a,b")-T2("i,j,a,b");
        double T2_change = delta_t2("i,j,a,b")*delta_t2("i,j,a,b");

        if (settings::rank == 0)
            printf("  @CCSD iteration %2d :  E(CCSD) =%16.10f  dE = %12.5e   RMS(T1) = %12.5e   RMS(T2) = %12.5e\n",
                   iter++, Eccn, Eccn - e_ccsd, std::sqrt(T1_change), std::sqrt(T2_change));

        // If converged, print final results
        if (std::fabs(Eccn - e_ccsd) < 1.0e-10 & std::sqrt(T1_change) < 1.0e-10 & std::sqrt(T2_change) < 1.0e-10) {
            converged = true;
            print("  CCSD has converged!\n  CCSD correlation energy:      %20.14lf\n", Eccn );
        }
        // If not converged, update Energy and Amplitudes
        e_ccsd = Eccn;
        T1 = t1n;
        T2 = t2n;

        ambit::timer::timer_pop();

        if (iter > maxiter) {
            printf("  CCSD has not converged in %d iterations!\n", maxiter);
            break;
        }
    } while (!converged);


    // CCSD(T) Spin-orbital Version
    //http://sirius.chem.vt.edu/wiki/doku.php?id=crawdad:programming:project6

    // build t3 denominators
//    Tensor Dijkabc = build("Dijkabc",{ndocc,ndocc,ndocc,nvocc,nvocc,nvocc});
//    Dijkabc.iterate([&](const std::vector<size_t>& indices, double& value) {
//        value = 1.0/(t_eigev.data()[indices[0]]+t_eigev.data()[indices[1]]+t_eigev.data()[indices[2]]
//                -t_eigev.data()[indices[3]+ndocc]-t_eigev.data()[indices[4]+ndocc]-t_eigev.data()[indices[5]+ndocc] );
//    });

//    //convert every needed thing to Spin-orbital version
//    Tensor t1_so = build("t1 in SO",{2*ndocc,2*nvocc});
//    t1_so("i,j") = T1("i,j");
//    Tensor t2_so = build("t2 in SO",{2*ndocc,2*nvocc});

//    Tensor T3Dc = build("Dijkabc Tijkabc (c) ",{ndocc,ndocc,ndocc,nvocc,nvocc,nvocc});




//    C.print(stdout, true);
//    C.iterate([](const std::vector<size_t>& /*indices*/, double& value) {
//        value += 1.0;
//    });
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
    ambit::settings::timers = true;
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
    ccsd();

    ambit::finalize();
    return EXIT_SUCCESS;
}

