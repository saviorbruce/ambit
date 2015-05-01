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
#include <ambit/blocked_tensor.h>

using namespace ambit;

TensorType tensor_type = kCore;

Tensor build(const std::string& name, const Dimension& dims)
{
    return Tensor::build(tensor_type, name, dims);
}

BlockedTensor buildblock(const std::string& name, const std::vector<std::string> &blocks)
{
    return BlockedTensor::build(tensor_type, name, blocks);
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

    Tensor C_all = build("C", AO2);
    C_all("i,j") = Smhalf("k,j") * Feigen["eigenvectors"]("i,k");
//    C.print(stdout, true);

    size_t ndocc = (size_t)n_docc;
    size_t nall = (size_t)nso;
    size_t nvocc = nall - ndocc;

    Tensor Cdocc = build("C", {ndocc, (size_t)nso});
    IndexRange CtoCdocc = { {0,ndocc}, {0,(size_t)nso}};
    //Cdocc.slice(C, CtoCdocc, CtoCdocc);
    Cdocc(CtoCdocc) = C_all(CtoCdocc);

    // Form initial D
    Tensor D = build("D", AO2);
    D("mu,nu") = Cdocc("i,mu") * Cdocc("i,nu");

    Tensor F = build("F", AO2);

    // start SCF iteration
    bool converged = false;
    double Eelec = 0.0, Eold = 0.0;
    int iter = 1, maxiter = 100;
    // For DIIS use
//    size_t diis_max = 4;
//    std::vector<Tensor> diis_F;
//    std::vector<Tensor> diis_E;
//    Tensor diis_C = build("diis_C",{diis_max});
    do {
        ambit::timer::timer_push("HF iteration");

        F("mu,nu") = H("mu,nu");
        F("mu,nu") += D("rho,sigma") * g2_g("mu,nu,rho,sigma");

//        F.print(stdout, true);

        // Calculate energy
        Eelec = D("mu,nu") * (H("mu,nu") + F("mu,nu"));



        // Transform the Fock matrix
        Ft("i,j") = Smhalf("mu,i") * Smhalf("nu,j") * F("mu,nu");

        // Diagonalize Fock matrix
        Feigen = Ft.syev(kAscending);

        // Construct new SCF eigenvector matrix.
        C_all("i,j") = Smhalf("k,j") * Feigen["eigenvectors"]("i,k");

        // Form new density matrix
        Cdocc(CtoCdocc) = C_all(CtoCdocc);
        Tensor D_new = build("D new", AO2);
        D_new("mu,nu") = Cdocc("i,mu") * Cdocc("i,nu");

        // Compute RMS of D matrix
        Tensor delta_D = build("Difference in D matrix", AO2);
        delta_D("mu,nu") = D_new("mu,nu") - D("mu,nu");
        double D_change = delta_D("mu,nu") * delta_D("mu,nu");

        print("  @RHF iter %5d: Escf = %17.12f  dE = %12.5e  RMS(D) = %12.5e\n", iter, Enuc + Eelec,Eelec-Eold,std::sqrt(D_change));

        if (std::fabs(Eelec - Eold) < 1.0e-13 && std::sqrt(D_change) < 1.0e-10) {
            converged = true;
            print("  HF has converged!\n  Final HF energy:        %20.14lf\n", Enuc + Eelec );
        }
        else {
            Eold = Eelec;
            D = D_new;
        }

//        // DIIS Extrapolation
//        // Error Matrix err_M = FDS- SDF for current iteration
//        Tensor err_M = build("error matrix",{nall,nall});
//        Tensor tmp = build("diis tmp matrix",{nall,nall});
//        tmp("r,s") = D("r,p")*S("p,s");
//        err_M("p,s") = F("p,r")*tmp("r,s");
//        tmp("r,s") = D("r,p")*F("p,s");
//        err_M("p,s") -= S("p,r")*tmp("r,s");

//        if(iter <= diis_max) {
//            diis_F.push_back(F);
//            diis_E.push_back(err_M);
//        }
//        else {
//            Tensor diis_B = build("B matrix",{diis_max+1,diis_max+1});
//            diis_B.iterate([&](const std::vector<size_t>& indices, double& value) {
//                if (indices[0]<diis_max && indices[1]<diis_max)
//                    value = diis_E[indices[0]]("i,j")*diis_E[indices[1]]("i,j")*;
//                else value = -1.0;
//                if (indices[0]==diis_max && indices[1]==diis_max) value = 0.0;
//            });
//            // need a lapack solver function.




//            int diis_i = iter%diis_max - 1;



//            print("stored %d F,  %d  \n", diis_F.size(),iter%diis_max);
//        }


        ambit::timer::timer_pop();

        if (iter++ > maxiter) {
            print("  HF has not converged in %d iterations!\n", maxiter);
            break;
        }
    } while (!converged);

//    MP2

    double e_scf = Enuc + Eelec;


    //prepare Cv matrix to use together with Co (Cdocc)

    Tensor Cvocc = build("Cv", {nvocc, nall});
    Cvocc({{0,nvocc},{0,nall}}) = C_all({{ndocc,nall},{0,nall}});

    // Blocked TEI Tensors
    std::vector<size_t> nso_list;
    std::vector<size_t> ndo_list;
    std::vector<size_t> nvo_list;
    for (size_t i=0;i<nall;++i) {
        nso_list.push_back(i);
        if (i<ndocc) ndo_list.push_back(i);
        else nvo_list.push_back(i);
    }

    BlockedTensor::add_mo_space("o","i,j,k,l,m,n",ndo_list,AlphaSpin);
    BlockedTensor::add_mo_space("v","a,b,c,d,e,f",nvo_list,AlphaSpin);
    BlockedTensor::add_composite_mo_space("g","p,q,r,s",{"o","v"});
    BlockedTensor::add_mo_space("h","w,x,y,z",nso_list,AlphaSpin);


    
    ambit::timer::timer_push("Integral Transformation");
    BlockedTensor G = buildblock("G",{"gggg"});
    {
        BlockedTensor C = buildblock("C",{"oh","vh"});
        C.block("vh")("pq") = Cvocc("pq");
        C.block("oh")("pq") = Cdocc("pq");
        BlockedTensor Gao = buildblock("AO basis G", {"hhhh"});
        Gao.block("hhhh")("pqrs") = g("pqrs");
        BlockedTensor Gtmp = buildblock("intermediate G_tmp",{"ghhh","hghh"});
        Gtmp("pxyz") = C("pw")*Gao("wxyz");
//        Gao.~BlockedTensor();
        Gtmp("xpyz") = Gtmp("pxyz");
        BlockedTensor Gtmp2 = buildblock("intermediate G_tmp2",{"gghh"});
        Gtmp2("qpyz") = C("qx")*Gtmp("xpyz");
        BlockedTensor Gtmp3 = buildblock("intermediate G_tmp3",{"gghg","gggh"});
        Gtmp3("qpyr") = C("rz")*Gtmp2("qpyz");
        Gtmp3("qpry") = Gtmp3("qpyr");
        BlockedTensor G_c = buildblock("G_c",{"gggg"});
        G_c("qprs") = C("sy")*Gtmp3("qpry");
        G("pqrs") = G_c("prqs");  // <pq|rs> = (pr|qs)
    }

    ambit::timer::timer_pop();


    // the energy eigenvalues

    Tensor t_eigev = Tensor::build(kCore, "eigenvalues", {nall});
    IndexRange all = {{0L, nall}};
    t_eigev(all) = Feigen["eigenvalues"](all);
//    std::vector<double> e_eigev = t_eigev.data();


//     Construct denominators
//     We only need D_IA and D_IjAb in Spin-adapted version

    ambit::timer::timer_push("Constructing denominators");

    Tensor Dia_o = build("Dia",{ndocc,nvocc});
    Dia_o.iterate([&](const std::vector<size_t>& indices, double& value) {
//        value = 1.0/(e_eigev[indices[0]]-e_eigev[indices[1]+ndocc]);
        value = 1.0/(t_eigev.data()[indices[0]]-t_eigev.data()[indices[1]+ndocc]);
    });

    Tensor Dijab_o = build("Dijab",{ndocc,ndocc,nvocc,nvocc});
    Dijab_o.iterate([&](const std::vector<size_t>& indices, double& value) {
        value = 1.0/(t_eigev.data()[indices[0]]+t_eigev.data()[indices[1]]
                -t_eigev.data()[indices[2]+ndocc]-t_eigev.data()[indices[3]+ndocc]);
    });
    
    BlockedTensor Dia = buildblock("Dia",{"ov"});
    Dia.block("ov")("pq") = Dia_o("pq");
    BlockedTensor Dijab = buildblock("Dijab",{"oovv"});
    Dijab.block("oovv")("pqrs") = Dijab_o("pqrs");

    ambit::timer::timer_pop();

    /* Close-shell CCSD */



    //Prepare the MO Basis Fock Matrix
//    Tensor Fmo = build("MO basis Fock matrix",{nall,nall});
//    Fmo("p,q") = F("mu,nu") * C("p,mu") * C("q,nu");
//    Fmo.print(stdout, true);

    // Build the Initial-Guess Cluster Amplitudes
    //spin-adapted version: we only need the TIA and TIjAb amplitudes
    //the initial TIJAB = TIjAb-TJiAb;
    BlockedTensor T1 = buildblock("T1",{"ov"});
    BlockedTensor T2 = buildblock("T2",{"oovv"});
    T2("ijab") = G("ijab")*Dijab("ijab");

    // Test the MP2 energy

    BlockedTensor T2_2 = buildblock("T2_2",{"oovv"});
    T2_2("ijab") = 2.0*T2("ijab")-T2("jiab");
    double e_mp2 = T2_2("ijab")*G("ijab");
//    double e_mp2 = G("ijab")*(2*T2("ijab")-T2("jiab"));
    print("T2[0][0][0][0]=%lf \n",T2.block("oovv").data()[0]);
    print("  MP2 Correlation Energy: %20.14lf\n", e_mp2);
    print("  Total MP2 Energy:       %20.14lf\n\n", e_scf+e_mp2);


//     Start CC iteration
    double e_ccsd = 0.0;
    converged = false;
    iter = 1, maxiter = 50;
    
    // Define all the tensors outside the loop
    BlockedTensor Tau_t = buildblock("Tau tilde",{"oovv"});
    BlockedTensor Tau = buildblock("Tau",{"oovv"});
    BlockedTensor Tau_tt = buildblock("Tau_tt",{"oovv","ovov"});
    BlockedTensor inter_f = buildblock("intermediate F",{"vv","oo","ov"});
    BlockedTensor inter_w = buildblock("intermediate W",{"oooo","vvvv","ovvo","vovo"});
//    BlockedTensor inter_w2 = buildblock("intermediate WMbeJ",{"ovvo"});
    BlockedTensor t1n = buildblock("T1 new",{"ov"});
    BlockedTensor T1n = buildblock("copy of T1 new",{"ov"});
    BlockedTensor sum_tmp = buildblock("tmp for sum",{"oo","vv"});
    BlockedTensor t2n = buildblock("T2 new",{"oovv"});
//    BlockedTensor T2n = buildblock("copy of t2n",{"oovv"});
//    BlockedTensor Tau_new = buildblock("new Tau",{"oovv"});
//    BlockedTensor Tau_new2_2 = buildblock("new Tau2_2",{"oovv"});
    BlockedTensor delta_t1 = buildblock("difference in T1",{"ov"});
    BlockedTensor delta_t2 = buildblock("difference in T2",{"oovv"});

    BlockedTensor G_k = buildblock("G_k",{"ovvv","ovvo","ooov","oovo","vovv","vovo","oovv"});
    G_k("maef") = G("amef");
    G_k("nfem") = G("mnef");
    G_k("mnje") = G("nmje");
    G_k("mnej") = G("nmej");
    G_k("eiab") = G("ieab");
    G_k("ejam") = G("ejma");
    G_k("ijba") = G("ijab");
    BlockedTensor w_tmp = buildblock("Wtmp",{"oooo","vvvv","ovvo","ovov","vovo"});
    BlockedTensor T2_k = buildblock("T2_k",{"ovov","oovv"});
    BlockedTensor Tau_tt_k = buildblock("Tau_tt_k",{"ovov"});
    BlockedTensor t_tmp = buildblock("t_tmp",{"oovv","ovov","vvvo","vvov","vovv","voov"});
    BlockedTensor t2_tmp = buildblock("t2_tmp",{"ovov","vvvo","vvov","vovv"});
    BlockedTensor T2_2_k = buildblock("T2_2_k",{"ovov"});

//    BlockedTensor G2_g = buildblock("2Gpqrs-Gqprs",{"oovv","vovv"});
////    G2_g("ijab") = 2.0*G("ijab")-G("jiab");
//    G2_g("mnef") = 2.0*G("mnef")-G("nmef");
//    G2_g("ciab") = 2.0*G("ciab")-G("icab"); // There is a bug here?
    BlockedTensor G2_g = buildblock("2Gpqrs-Gqprs",{"gggg"});
    G2_g("pqrs") = 2.0*G("pqrs") - G("qprs");
//    G2_g("aibc") = 2.0*G("aibc") - G("iabc"); //changed value -- bug?
//    G2_g("ciab") = 2.0*G("ciab")-G("icab"); // There is a bug here?
    BlockedTensor G2_g_k = buildblock("G2_g_k",{"ovvo","ovov","vvov","ooov"});
    G2_g_k("nfem") = G2_g("mnef");
    G2_g_k("nfme") = G2_g("mnef");
    G2_g_k("aemf") = G2_g("amef");
    G2_g_k("mine") = G2_g("mnie");
    BlockedTensor G2_g_k2 = buildblock("G2_g_k2",{"ovov"});
    G2_g_k2("iame") = G2_g("amie");

    // initial Tau --- updated at the end of each iteration
    t_tmp("iajb") = T1("ia")*T1("jb");
    Tau("ijab") = T2("ijab");
    Tau("ijab") += t_tmp("iajb");
    Tau_t("ijab") = T2("ijab");
    Tau_t("ijab") += 0.5*t_tmp("iajb");
    Tau_tt("ijab") = T2("ijab");
    Tau_tt("ijab") += 2.0*t_tmp("iajb");

    
    do {

        ambit::timer::timer_push("CCSD iteration");

        //Tau_tilde, Tau, and Tau_tt
//        BlockedTensor Tau_t = buildblock("Tau tilde",{"oovv"});
//        Tau_t("ijab") = T2("ijab");
//        Tau_t("ijab") += 0.5*T1("ia")*T1("jb");
//        BlockedTensor Tau = buildblock("Tau",{"oovv"});
//        Tau("ijab") = T2("ijab");
//        Tau("ijab") += T1("ia")*T1("jb");
//        BlockedTensor Tau_tt = buildblock("Tau_tt",{"oovv"});
//        Tau_tt("ijab") = T2("ijab");
//        Tau_tt("ijab") += 2.0*T1("ia")*T1("jb");



        //Form the intermediates
        // intermediate F matrix
//        BlockedTensor inter_f = buildblock("intermediate F",{"vv","oo","ov"});
//        inter_f("ae") = T1("mf")*(2*G("amef")-G("maef"));
        inter_f("ae") = T1("mf")*G2_g_k("aemf");
//        inter_f("ae") -= Tau_t("mnaf")*(2*G("mnef")-G("nmef"));
        inter_f("ae") -= Tau_t("nmfa")*G2_g("nmfe");

//        inter_f("mi") = T1("ne")*(2*G("mnie")-G("nmie"));
        inter_f("mi") = T1("ne")*G2_g_k("mine");
//        inter_f("mi") += Tau_t("inef")*(2*G("mnef")-G("nmef"));
        inter_f("mi") += Tau_t("inef")*G2_g("mnef");

//        inter_f("me") = T1("nf")*(2*G("mnef")-G("nmef"));
        inter_f("me") = T1("nf")*G2_g_k("nfme");

//        BlockedTensor inter_w = buildblock("intermediate W",{"oooo","vvvv","ovvo"});

        inter_w("mnij") = G("mnij");
        inter_w("mnij") += T1("je")*G("mnie");
//        inter_w("mnij") += T1("ie")*G("nmje");
        w_tmp("mnji") = T1("ie")*G_k("mnje");
        inter_w("mnij") += w_tmp("mnji");
        inter_w("mnij") += 0.5*Tau("ijef")*G("mnef");

//        inter_w("abef") = G("abef"); // take 1.1s?
//        inter_w("abef") -= T1("mb")*G("amef");    // This takes 6s total
//        inter_w("abef") -= T1("mb")*G_k("maef");  // This still takes 6s
//        BlockedTensor w_tmp = buildblock("Wtmp",{"vvvv"});
//        inter_w.block("vvvv").zero(); // 0.2s
        w_tmp("baef") = -T1("mb")*G_k("maef"); // This take 0.7s total
        inter_w("abef") = w_tmp("baef");  // 0.9s when += , 1.1s when =
        inter_w("abef") -= T1("ma")*G("mbef");
        inter_w("abef") += 0.5*Tau("mnab")*G("mnef");
//        inter_w("abef") += G("abef"); // take 0.8s

        inter_w("mbej") = G("mbej");
        inter_w("mbej") += T1("jf")*G("mbef");
//        inter_w("mbej") -= T1("nb")*G("nmje");
        w_tmp("bmej") = -T1("nb")*G_k("nmej");
        inter_w("mbej") += w_tmp("bmej");
//        inter_w("mbej") -= 0.5*Tau_tt("njbf")*G("mnef");
        Tau_tt("jbnf") = Tau_tt("jnfb");
        w_tmp("jbem") = -0.5*Tau_tt("jbnf")*G_k("nfem");
//        inter_w("mbej") += 0.5*T2("jnbf")*(2*G("mnef")-G("nmef"));
        T2_k("jbnf") = T2("jnbf");
        w_tmp("jbem") += 0.5*T2_k("jbnf")*G2_g_k("nfem");
        inter_w("mbej") += w_tmp("jbem");

        //inter_w abba case        
//        BlockedTensor inter_w2 = buildblock("intermediate WMbeJ",{"ovvo"});
//        inter_w2("mbej") = -G("mbje");
//        inter_w2("mbej") -= T1("jf")*G("bmef"); // This is much faster than G("mbfe")!
//        inter_w2("mbej") += T1("nb")*G("mnje");
//        inter_w2("mbej") += 0.5*Tau_tt("jnfb")*G("nmef");
        inter_w("bmej") = -G("bmej");
        inter_w("bmej") -= T1("jf")*G("bmef");
        inter_w("bmej") += T1("nb")*G("nmej");
//        inter_w("bmej") += 0.5*Tau_tt("jnfb")*G("nmef");
        Tau_tt_k("jbnf") = Tau_tt("jnfb");
        w_tmp("jbem") = 0.5*Tau_tt_k("jbnf")*G("nfem");
        inter_w("bmej") += w_tmp("jbem");
//        inter_w2("mbej") = inter_w("bmej");



        //Compute new T1 and T2 amplitudes
        //new T1
//        BlockedTensor t1n = buildblock("T1 new",{"ov"});
        t1n("ia") = T1("ie")*inter_f("ae");
        t1n("ia") -= T1("ma")*inter_f("mi");
//        t1n("ia") += inter_f("me")*(2*T2("imae")-T2("miae"));
        T2_2("imae") = 2.0*T2("imae")-T2("miae");
        t_tmp("iame") = T2_2("imae");
        t1n("ia") += inter_f("me")*t_tmp("iame");
//        t1n("ia") += T1("me")*(2*G("imae")-G("maie"));
        t1n("ia") += T1("me")*G2_g_k2("iame");
//        t1n("ia") -= T2("mnae")*(2*G("mnie")-G("nmie"));
        t1n("ia") -= T2("nmea")*G2_g("nmei");
//        t1n("ia") += T2("imef")*(2*G("amef")-G("maef"));
        t1n("ia") += T2("imef")*G2_g("amef");
//        BlockedTensor T1n = buildblock("copy of T1 new",{"ov"});
        T1n("ia") =  t1n("ia");
        t1n("ia") =  T1n("ia")*Dia("ia");

        //new T2
//        BlockedTensor t2n = buildblock("T2 new",{"oovv"});
//        BlockedTensor sum_tmp = buildblock("tmp for sum",{"oo","vv"});
        t2n("ijab") = G("ijab");
        sum_tmp("be") = inter_f("be");
        sum_tmp("be") -= 0.5*T1("mb")*inter_f("me");
        t2n("ijab") += T2("ijae")*sum_tmp("be");
//        t2n("ijab") += T2("ijeb")*sum_tmp("ae");
        T2_k("ijbe") = T2("ijeb");
        t_tmp("ijba") = T2_k("ijbe")*sum_tmp("ae");
        t2n("ijab") += t_tmp("ijba");
        sum_tmp("mj") = inter_f("mj");
        sum_tmp("mj") += 0.5*T1("je")*inter_f("me");
//        t2n("ijab") -= T2("imab")*sum_tmp("mj");
        t_tmp("jiab") = T2_k("miab")*sum_tmp("mj");
        t2n("ijab") -= t_tmp("jiab");
        t2n("ijab") -= T2("mjab")*sum_tmp("mi");
        t2n("ijab") += Tau("mnab")*inter_w("mnij");

        t2n("ijab") += Tau("ijef")*G("abef");  // 0.6s
        t2n("ijab") += Tau("ijef")*inter_w("abef");
        // (1)
//        T2_2("imae") = 2.0*T2("imae")-T2("miae");
        T2_2_k("iame") = T2_2("imae");
        w_tmp("mejb") = inter_w("mbej");
        t_tmp("iajb") = T2_2_k("iame")*w_tmp("mejb");
        // (2)
//        T2_k("iame") = T2("imae");
        w_tmp("mejb") = inter_w("bmej");
        t_tmp("iajb") += T2_k("iame")*w_tmp("mejb");
        // (3)
        T2_k("ibme") = T2("mibe");
//        w_tmp("meja") = inter_w("amej");
        t2_tmp("ibja") = T2_k("ibme")*w_tmp("meja");
        t_tmp("iajb") += t2_tmp("ibja");
        // (4)
        t2_tmp("ajeb") = -T1("ma")*G("mjeb");
        t2_tmp("eajb") = t2_tmp("ajeb");
//        t_tmp("ejba") = -T1("ma")*G_k("ejbm");
//        t2_tmp("ejab") = t_tmp("ejba");   // This is even slower..
        // (5)
        t2_tmp("ejab") = -T1("mb")*G("ejam");
//        t2_tmp("ejab") -= T1("mb")*G("ejam");
        t2_tmp("eajb") += t2_tmp("ejab");
        t_tmp("iajb") += T1("ie")*t2_tmp("eajb"); // for both (4) and (5)
        t2n("ijab") += t_tmp("iajb");  // account for (1) --(5)
        t2n("ijab") += t_tmp("jbia");  // account for (1)'--(5)'


//        t2n("ijab") += (T2("imae")-T2("miae"))*inter_w("mbej");  // 1
//        t2n("ijab") -= T1("ie")*T1("ma")*G("mjeb");
//        t2n("ijab") += T2("imae")*inter_w("mbej");  // 1
//        t2n("ijab") += T2("imae")*inter_w("bmej");

//        t2n("ijab") += T2("mibe")*inter_w("amej");
//        t2n("ijab") -= T1("ie")*T1("mb")*G("maje");
//        t2n("ijab") += T2("mjae")*inter_w("bmei");
//        t2n("ijab") -= T1("je")*T1("ma")*G("mbie");
//        t2n("ijab") += (T2("jmbe")-T2("mjbe"))*inter_w("maei");  // 1
//        t2n("ijab") -= T1("je")*T1("mb")*G("miea");
//        t2n("ijab") += T2("jmbe")*inter_w("maei");  // 1
//        t2n("ijab") += T2("jmbe")*inter_w("amei");

        t2n("ijab") += T1("ie")*G("ejab");
//        t2n("ijab") += T1("je")*G("abie");
        t_tmp("jiab") = T1("je")*G_k("eiab");
        t2n("ijab") += t_tmp("jiab");
//        t2n("ijab") -= T1("ma")*G("ijmb"); // 40ms
        t_tmp("ajib") = -T1("ma")*G("mjib"); // 6ms
        t2n("ijab") += t_tmp("ajib");  // 4ms

        t2n("ijab") -= T1("mb")*G("ijam");
//        BlockedTensor T2n = buildblock("copy of t2n",{"oovv"});
        t_tmp("ijab") = t2n("ijab");
        t2n("ijab") = t_tmp("ijab") * Dijab("ijab");
//        t2n.print(stdout,true);

        //compute new CCSD energy
//        BlockedTensor Tau_new = buildblock("new Tau",{"oovv"});
//        Tau_new("ijab") = t2n("ijab");
//        Tau_new("ijab") += t1n("ia")*t1n("jb");
        t_tmp("iajb") = t1n("ia")*t1n("jb");
        Tau("ijab") = t2n("ijab");
        Tau("ijab") += t_tmp("iajb");
        Tau_t("ijab") = t2n("ijab");
        Tau_t("ijab") += 0.5*t_tmp("iajb");
        Tau_tt("ijab") = t2n("ijab");
        Tau_tt("ijab") += 2.0*t_tmp("iajb");
        double Eccn = Tau("ijab")*G2_g("ijab");
        //compute RMS in T1 and T2 amplitudes
//        BlockedTensor delta_t1 = buildblock("difference in T1",{"ov"});
        delta_t1("ia") = t1n("ia")-T1("ia");
        double T1_change = delta_t1("ia")*delta_t1("ia");
//        BlockedTensor delta_t2 = buildblock("difference in T2",{"oovv"});
        delta_t2("ijab") = t2n("ijab")-T2("ijab");
        double T2_change = delta_t2("ijab")*delta_t2("ijab");

        if (settings::rank == 0)
            print("  @CCSD iteration %2d :  E(CCSD) =%16.10f  dE = %12.5e   RMS(T1) = %12.5e   RMS(T2) = %12.5e\n",
                   iter++, Eccn, Eccn - e_ccsd, std::sqrt(T1_change), std::sqrt(T2_change));

        // If converged, print final results
        if (std::fabs(Eccn - e_ccsd) < 1.0e-12 & std::sqrt(T1_change) < 1.0e-7 & std::sqrt(T2_change) < 1.0e-7) {
            converged = true;
            print("  CCSD has converged!\n  CCSD correlation energy:      %20.14lf\n", Eccn );
        }
        // If not converged, update Energy and Amplitudes
        e_ccsd = Eccn;
        T1("ia") = t1n("ia");
        T2("ijab") = t2n("ijab");

        ambit::timer::timer_pop();

        if (iter > maxiter) {
            print("  CCSD has not converged in %d iterations!\n", maxiter);
            break;
        }
    } while (!converged);

}

int main(int argc, char* argv[])
{
    srand(time(nullptr));
    ambit::settings::timers = true;
    ambit::initialize(argc, argv);

    if (argc > 1) {
        if (settings::distributed_capable && strcmp(argv[1], "cyclops") == 0) {
            tensor_type = kDistributed;
            print("  *** Testing distributed tensors. ***\n");
        }
        else {
            print("  *** Unknown parameter given ***\n");
            print("  *** Testing core tensors.   ***\n");
        }
    }
    ccsd();

    ambit::finalize();
    return EXIT_SUCCESS;
}

