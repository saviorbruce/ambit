#include <math.h>
#include "twocenter.h"

namespace libgaussian {

TwoCenterOverlapInt::TwoCenterOverlapInt(
    std::shared_ptr<SBasisSet> basis1,
    std::shared_ptr<SBasisSet> basis2,
    int deriv) :
    TwoCenterInt(basis1,basis2)
{
    size_t chunk_size = basis1->max_ncartesian() * basis2->max_cartesian();
    data1_.resize(1L * chunk_size);
    data2_.resize(1L * chunk_size);
}
TwoCenterOverlapInt::compute_shell0(
    const SGaussianShell& sh1,
    const SGaussianShell& sh2)
{
    int L1 = sh1.am();
    int L2 = sh2.am();

    int N1 = sh1.ncartesian();
    int N2 = sh2.ncartesian();
    
    int K1 = sh1.nprimitive();
    int K2 = sh2.nprimitive();

    const std::vector<double>& cs1 = sh1.cs();
    const std::vector<double>& cs2 = sh2.cs();

    const std::vector<double>& es1 = sh1.es();
    const std::vector<double>& es2 = sh2.es();

    double x1 = sh1.x();
    double y1 = sh1.y();
    double z1 = sh1.z();

    double x2 = sh2.x();
    double y2 = sh2.y();
    double z2 = sh2.z();

    double x12 = x1 - x2; 
    double y12 = y1 - y2; 
    double z12 = z1 - z2; 

    double R1[(L1 + 1)];
    double R2[(L2 + 1)];

    double Sx[(L1 + 1) * (L2 + 1)];    
    double Sy[(L1 + 1) * (L2 + 1)];    
    double Sz[(L1 + 1) * (L2 + 1)];    

    double* data1p = data1.data();

    for (int k1 = 0; k1 < K1; k1++) {
        double c1 = cs1[k1];
        double e1 = es1[k1];
        for (int k2 = 0; k2 < K2; k2++) {
            double c2 = cs2[k2];
            double e2 = es2[k2];

            double p = e1 + e2;
            double pm1 = 1.0 / p;
            double pm12 = pow(p,-1.0/2.0);
            double mu = e1 * e2 * pm1;
            double xP = (e1 * x1 + e2 * x2) * pm1;
            double yP = (e1 * y1 + e2 * y2) * pm1;
            double zP = (e1 * z1 + e2 * z2) * pm1;
            double Kx = exp(- mu * x12 * x12);
            double Ky = exp(- mu * y12 * y12);
            double Kz = exp(- mu * z12 * z12);

            Kz *= c1 * c2; 

            double xP1 = xP - x1;
            double xP2 = xP - x2;
            memset(Sx,'\0',(L1 + 1) * (L2 + 1));
            for (int i = 0; i < nquad; i++) {
                double xq1 = xq[i] * pm12 + xP1;
                R1[0] = wq[i] * Kx * pm12;
                for (int l = 1; l <= L1; l++) {
                    R1[l] = R1[l-1] * xq1;
                }
                double xq2 = xq[i] * pm12 + xP2;
                R2[0] = 1.0;
                for (int l = 1; l <= L2; l++) {
                    R2[l] = R2[l-1] * xq2;
                }
                for (int l1 = 0; l1 <= L1; l1++) {
                    for (int l2 = 0; l2 <= L2; l2++) {
                        Sx[l1 * (L2 + 1) + l2] += R1[l1] * R2[l2]; 
                    }
                }
            }

            double yP1 = yP - y1;
            double yP2 = yP - y2;
            memset(Sy,'\0',(L1 + 1) * (L2 + 1));
            for (int i = 0; i < nquad; i++) {
                double yq1 = yq[i] * pm12 + yP1;
                R1[0] = wq[i] * Ky * pm12;
                for (int l = 1; l <= L1; l++) {
                    R1[l] = R1[l-1] * yq1;
                }
                double yq2 = yq[i] * pm12 + yP2;
                R2[0] = 1.0;
                for (int l = 1; l <= L2; l++) {
                    R2[l] = R2[l-1] * yq2;
                }
                for (int l1 = 0; l1 <= L1; l1++) {
                    for (int l2 = 0; l2 <= L2; l2++) {
                        Sy[l1 * (L2 + 1) + l2] += R1[l1] * R2[l2]; 
                    }
                }
            }

            double zP1 = zP - z1;
            double zP2 = zP - z2;
            memset(Sz,'\0',(L1 + 1) * (L2 + 1));
            for (int i = 0; i < nquad; i++) {
                double zq1 = zq[i] * pm12 + zP1;
                R1[0] = wq[i] * Kz * pm12;
                for (int l = 1; l <= L1; l++) {
                    R1[l] = R1[l-1] * zq1;
                }
                double zq2 = zq[i] * pm12 + zP2;
                R2[0] = 1.0;
                for (int l = 1; l <= L2; l++) {
                    R2[l] = R2[l-1] * zq2;
                }
                for (int l1 = 0; l1 <= L1; l1++) {
                    for (int l2 = 0; l2 <= L2; l2++) {
                        Sz[l1 * (L2 + 1) + l2] += R1[l1] * R2[l2]; 
                    }
                }
            }
             
            double* Ip = data1p;
            for (int i1 = 0; i1 <= L1; i1++) {
                int l1 = L1 - i1;
                for (int j1 = 0; j1 <= i1; j1++) {
                    int m1 = i1 - j1;
                    int n1 = j1;
                    for (int i2 = 0; i2 <= L2; i2++) {
                        int l2 = L2 - i2;
                        for (int j2 = 0; j2 <= i2; j2++) {
                            int m2 = i2 - j2;
                            int n2 = j2;
                            (*Ip++) += 
                                Sx[l1 * (L1 + 1) + l2] *
                                Sy[m1 * (L1 + 1) + m2] *
                                Sz[n1 * (L1 + 1) + n2];
                        }
                    }
                }
            }
        }
    }

}


} // namespace libgaussian
