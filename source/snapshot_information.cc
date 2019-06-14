/*
 * snapshot_information.cc
 *
 *  Created on: Jun 14, 2019
 *      Author: sg
 */

#include "snapshot_information.h"

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>

#include <boost/serialization/utility.hpp>
#include <boost/serialization/map.hpp>

#include <cassert>

namespace Snapshot {

SnapshotInformation::SnapshotInformation
(const unsigned int  timestep_number,
 const double        time,
 const double        timestep,
 const double        old_timestep)
:
timestep_number_(timestep_number),
time_(time),
timesteps(timestep, old_timestep)
{}

void SnapshotInformation::set_parameters
(const double ekman,
 const double prandtl,
 const double rayleigh)
{
    parameters_["Ek"] = ekman;
    parameters_["Pr"] = prandtl;
    parameters_["Ra"] = rayleigh;
}
template<typename Stream>
void SnapshotInformation::print(Stream &stream)
{
    if (parameters_.empty())
        return;

    stream << "Snapshot information: " << std::endl;
    stream << "   Step: "
           << dealii::Utilities::int_to_string(timestep_number_, 8) << ", "
           << "time: " << time_
           << std::endl;
    stream << "   time step: " << timestep() << ", "
           << "old time step: " << old_timestep()
           << std::endl;

    stream << "Snapshot parameters: " << std::endl;

    stream << "   +----------+----------+----------+\n"
           << "   |    Ek    |    Ra    |    Pr    |\n"
           << "   +----------+----------+----------+\n"
           << "   | "
           << std::setw(8) << std::setprecision(1) << std::scientific << std::right << parameters_["Ek"]
           << " | "
           << std::setw(8) << std::setprecision(1) << std::scientific << std::right << parameters_["Pr"]
           << " | "
           << std::setw(8) << std::setprecision(1) << std::scientific << std::right << parameters_["Ra"]
           << " |\n"
           << "   +----------+----------+----------+\n";

    stream << std::endl << std::setprecision(6) << std::fixed << std::flush;
}

unsigned int SnapshotInformation::timestep_number() const
{
    return timestep_number_;
}

double SnapshotInformation::timestep() const
{
    return timesteps.first;
}

double SnapshotInformation::old_timestep() const
{
    return timesteps.second;
}

template<typename Archive>
void SnapshotInformation::serialize(Archive  &ar, const unsigned int /* version */)
{
    ar & timestep_number_;
    ar & time_;
    ar & timesteps;
    ar & parameters_;
}

void save(std::ofstream &os, const SnapshotInformation &snapshot_info)
{
    boost::archive::text_oarchive   oa{os};

    oa << snapshot_info;
}

void load(std::ifstream &is, SnapshotInformation &snapshot_info)
{
    boost::archive::text_iarchive   ia{is};

    ia >> snapshot_info;

    assert(snapshot_info.timestep() >= 0);
    assert(snapshot_info.old_timestep() >= 0);
}
}  // namespace Snapshot


// explicit instantiation
template void Snapshot::SnapshotInformation::print(dealii::ConditionalOStream &);
