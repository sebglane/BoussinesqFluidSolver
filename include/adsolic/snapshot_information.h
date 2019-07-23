/*
 * snapshot_info.h
 *
 *  Created on: Jun 14, 2019
 *      Author: sg
 */

#ifndef INCLUDE_SNAPSHOT_INFORMATION_H_
#define INCLUDE_SNAPSHOT_INFORMATION_H_

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <fstream>
#include <map>
#include <utility>

namespace Snapshot {

class SnapshotInformation
{
public:
    SnapshotInformation() = default;
    SnapshotInformation(const unsigned int  timestep_number,
                        const double        time,
                        const double        timestep,
                        const double        old_timestep);

    template<typename Stream>
    void    print(Stream &stream);
    void    set_parameters(const double ekman,
                           const double prandtl,
                           const double rayleigh);

    double  timestep() const;
    double  old_timestep() const;

    unsigned int timestep_number() const;

private:
    friend class boost::serialization::access;

    template<typename Archive>
    void serialize(Archive  &ar, const unsigned int version);

    unsigned int    timestep_number_;

    double  time_;

    std::pair<double,double>    timesteps;

    std::map<std::string,double>    parameters_;
};

void save(std::ofstream &os, const SnapshotInformation &snapshot_info);

void load(std::ifstream &is, SnapshotInformation &snapshot_info);

}  // namespace Snapshot

#endif /* INCLUDE_SNAPSHOT_INFORMATION_H_ */
