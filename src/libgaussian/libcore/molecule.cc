#include <math.h>
#include "molecule.h"

namespace libgaussian {

double SAtom::distance(const SAtom& other) const 
{
    double dx = x_ - other.x_;
    double dy = y_ - other.y_;
    double dz = z_ - other.z_;
    return sqrt(dx*dx + dy*dy + dz*dz);
}

SMolecule::SMolecule(
    const std::string& name,
    const std::vector<SAtom>& atoms) :
    name_(name),
    atoms_(atoms)
{
    for (size_t A = 0; A < atoms_.size(); A++) {
        atoms_[A].set_index(A);
    }
}
double SMolecule::nuclear_repulsion_energy() const 
{
    double E = 0.0;
    for (size_t A = 0; A < atoms_.size(); A++) {
        for (size_t B = A + 1; B < atoms_.size(); B++) {
            E += atoms_[A].Z() * atoms_[B].Z() / atoms_[A].distance(atoms_[B]);
        }
    }
    return E;
}

} // namespace libgaussian
