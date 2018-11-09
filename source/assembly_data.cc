/*
 * assembly_data.cc
 *
 *  Created on: Nov 7, 2018
 *      Author: sg
 */


#include "assembly_data.templates.h"

// explicit instantiation
template class TemperatureAssembly::Scratch::Matrix<2>;
template class TemperatureAssembly::Scratch::Matrix<3>;
template class TemperatureAssembly::Scratch::RightHandSide<2>;
template class TemperatureAssembly::Scratch::RightHandSide<3>;

template class TemperatureAssembly::CopyData::Matrix<2>;
template class TemperatureAssembly::CopyData::Matrix<3>;
template class TemperatureAssembly::CopyData::RightHandSide<2>;
template class TemperatureAssembly::CopyData::RightHandSide<3>;

template class StokesAssembly::Scratch::Matrix<2>;
template class StokesAssembly::Scratch::Matrix<3>;
template class StokesAssembly::Scratch::RightHandSide<2>;
template class StokesAssembly::Scratch::RightHandSide<3>;

template class StokesAssembly::CopyData::Matrix<2>;
template class StokesAssembly::CopyData::Matrix<3>;
template class StokesAssembly::CopyData::RightHandSide<2>;
template class StokesAssembly::CopyData::RightHandSide<3>;
