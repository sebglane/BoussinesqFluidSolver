/*! @file     doc-example_gravity_field.h
 *  @author   Max Mustermann
 *  @date     Jul 11, 2019
 *  @brief    Brief description of the .h file
 *  @details  Long description of the .h file
 *  @bug      There is a bug with ...
 *  @warning  Does not work if ...
 */

#ifndef INCLUDE_GRAVITY_FIELD_CC_
#define INCLUDE_GRAVITY_FIELD_CC_

#include <deal.II/base/tensor_function.h>

namespace EquationData {

/*!
 *  @brief A enum class for gravity models.
 *  This enum class indicates which gravity model is to be implemented.
 *  The built-in models include a constant gravity over the whole domain
 *  and a linear increasing gravity along one coordinate.
 */
enum GravityProfile
{
    Constant,               /*!< Constant gravity model. */
    Linear,                 /*!< Linear gravity model. */
    default_profile=Linear  /*!< Default is the linear model */
};

using namespace dealii;

/*! @brief A class defining the gravity vector.
 *  This enum class indicates which gravity model is to be implemented.
 *  The built-in models include a constant gravity over the whole domain
 *  and a linear increasing gravity along one coordinate.
 *  In the constant case the gravity tensor is filled with,
 *  @f[
 *      \mathbf{g} = \frac{\mathbf{x}}{\left\Vert \mathbf{x} \right\Vert}
 *  @f]
 *  and the linear case by
 *  @f[
 *      \mathbf{g} = \frac{s}{r_o}\mathbf{x}
 *  @f]
 *  where @f$ s @f$ and @f$ r_o @f$ are the scaling factor and the outer
 *  radius respectively.
 */
template <int dim>
class GravityFunction : public TensorFunction<1,dim>
{
public:
    
    /*! @brief A brief description of the GravitiyFuncition constructor 
     *  @details A detailed description of the constructor
     */
    GravityFunction(const double    outer_radius = 1.0,
                    const GravityProfile    profile_type = GravityProfile::Constant);
    
    /*! @brief A brief description of the value method 
     *  @details A detailed description of the value method
     */
    virtual Tensor<1,dim>   value(const Point<dim> &p) const;
    
    /*! @brief A brief description of the value_list method 
     *  @details A detailed description of the value_list method
     */
    virtual void            value_list(const std::vector<Point<dim>>    &points,
                                       std::vector<Tensor<1,dim>>       &values) const;
    
    /*! @brief A brief description of the get_profile_type method 
     *  @details A detailed description of the get_profile_type method
     */
    GravityProfile  get_profile_type() const;

private:
    const double outer_radius;                //!< Brief description of outer_radius.

    const double scaling_factor = 20. / 13.;  //!< Brief description of scaling_factor.

    const GravityProfile    profile_type;     //!< Brief description of profile_type.
};

}  // namespace EquationData

#endif /* INCLUDE_GRAVITY_FIELD_CC_ */
