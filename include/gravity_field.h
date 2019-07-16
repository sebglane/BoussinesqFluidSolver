/*
 * gravity_field.cc
 *
 *  Created on: Jul 11, 2019
 *      Author: sg
 */

#ifndef INCLUDE_GRAVITY_FIELD_CC_
#define INCLUDE_GRAVITY_FIELD_CC_

#include <deal.II/base/tensor_function.h>

namespace EquationData {

/*
 *
 * enumeration for the type of the gravity profile
 *
 */
enum GravityProfile
{
    ConstantRadial,
    LinearRadial,
    ConstantCartesian,
    default_profile=LinearRadial
};

using namespace dealii;

template <int dim>
class GravityFunction : public TensorFunction<1,dim>
{
public:
    GravityFunction(const double    outer_radius = 1.0,
                    const GravityProfile    profile_type = GravityProfile::ConstantRadial);

    virtual Tensor<1,dim>   value(const Point<dim> &p) const;

    virtual void            value_list(const std::vector<Point<dim>>    &points,
                                       std::vector<Tensor<1,dim>>       &values) const;

    GravityProfile  get_profile_type() const;

private:
    const double outer_radius;

    const double scaling_factor = 20. / 13.;

    const GravityProfile    profile_type;
};

}  // namespace EquationData

#endif /* INCLUDE_GRAVITY_FIELD_CC_ */
