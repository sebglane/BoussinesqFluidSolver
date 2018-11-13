#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import dolfin as dlfn
import mshr
from dolfin import inner, grad, div, dot
import numpy as np
#==============================================================================
# TODO: try to use MUMPS or UMFPACK instead of LUSolver
# TODO: more diagnostic values
# TODO: write output of sol00 sol0 for restart
# TODO: implement restart
#==============================================================================
#dlfn.parameters["form_compiler"]["cpp_optimize"] = True
#comm = dlfn.mpi_comm_world()
#mpi_comm = dlfn.mpi_comm_world()
#mpi_rank = dlfn.MPI.rank(mpi_comm)
#print mpi_rank
print "The system's version of FEniCS is ", dlfn.dolfin_version(), "."
#==============================================================================
# run time parameters
t_end = 150.
n_steps = 50000
output_frequency = 50
checkpoint_frequency = 1000
rms_frequency = 5
aspect_ratio = 0.35
print_time_stepping_coefficients = False
use_assembler_method = True
non_zero_initial_temperature = True
#==============================================================================
# refinements
n_initial_refinements = 0
#==============================================================================
# time stepping parameters
adaptive_time_stepping = True
dt_initial = 1e-6
cfl_min = 0.4
cfl_max = 0.6
dt_max = 5e-1
dt_min = 1e-12
#==============================================================================
# dimensionless constants
rotation = True
ekman = 1e-3
rayleigh = 1e5
prandtl = 1.0
#==============================================================================
pvd_velocity = dlfn.File("pvd/solution-velocity.pvd")
pvd_pressure = dlfn.File("pvd/solution-pressure.pvd")
pvd_temperature = dlfn.File("pvd/solution-temperature.pvd")
#==============================================================================
# equation coefficients
if rotation == True:
    tmp = (2.0 / ekman, 1.0, rayleigh / prandtl, 1.0 / prandtl)
else:
    tmp = (0.0, np.sqrt(prandtl/ rayleigh), 1.0, 1.0 / np.sqrt(rayleigh * prandtl) )
equation_coefficients = tuple([dlfn.Constant(t) for t in tmp])
#==============================================================================
print """------------------------------------------------------------------------
Boussinesq solver by S. Glane and B. Turan

This program solves the Navier-Stokes system with thermal convection. We use
a direct solver and the stable Taylor-Hood (P2-P1) element.

The governing equations are

\t- Incompressibility constraint:
\t\t div(v) = 0,
\t- Navier-Stokes equation:
\t\tdv/dt + v . grad(v) + C1 Omega .times. v 
\t\t\t\t= - grad(p) + C2 div(grad(v)) - C3 T g,

\t- Heat conduction equation:
\t\tdT/dt + v . grad(T) = C4 div(grad(T)).

The coefficients C1 to C4 depend on the normalization.

+-------------------+-------+---------------+---------+-------------------+
|   Normalization   |    C1 |      C2       |   C3    |        C4         |
+-------------------+-------+---------------+---------+-------------------+
| Non-rotating case | 0     | sqrt(Pr / Ra) | 1       | 1 / sqrt(Ra * Pr) |
| Rotating case     | Ek    | 1             | Ra / Pr | 1 / Pr            |
+-------------------+-------+---------------+---------+-------------------+"""
if rotation:
    print """You have chosen the rotating case with:
\t Ek = {0:3.2e},\t Ra = {1:3.2e},\t Pr = {2:3.2e} .""".format(ekman,rayleigh,prandtl)
else:
    print """You have chosen the non-rotating case with:
\t Ra = {0:3.2e},\t Pr = {1:3.2e} .""".format(rayleigh,prandtl)
    
print """
The related coefficients C1 to C4 are given by:
\t C1 = {0:3.2e},\t C2 = {1:3.2e},\t C3 = {2:3.2e},\t C4 = {3:3.2e} .\n""".format(
    *map(lambda x: x.values()[0], equation_coefficients))
#==============================================================================
# mesh creation
# inner / outer radius
outer_radius = 1. 
inner_radius = aspect_ratio * outer_radius
# dolfin mshr module
center = dlfn.Point(0., 0.)
domain = mshr.Circle(center, outer_radius) \
       - mshr.Circle(center, inner_radius)
mesh = mshr.generate_mesh(domain, 50)
for i in range(n_initial_refinements):
    mesh = dlfn.refine(mesh)
space_dim = mesh.topology().dim()
n_cells = mesh.num_cells()
#==============================================================================
# subdomains for boundaries
facet_marker = dlfn.MeshFunction("size_t", mesh, space_dim-1)
facet_marker.set_all(0)
# length of smalles element
hmin = mesh.hmin() 
# inner circle boundary
class InnerCircle(dlfn.SubDomain):
    def inside(self, x, on_boundary):
        tol = hmin/2. # tolerance: half length of smallest element
        result = abs(np.sqrt(x[0]**2 + x[1]**2) - inner_radius) < tol
        return result and on_boundary
# outer cirlce boundary
class OuterCircle(dlfn.SubDomain):
    def inside(self, x, on_boundary):
        tol = hmin/2. # tolerance: half length of smallest element
        result = abs(np.sqrt(x[0]**2 + x[1]**2) - outer_radius) < tol
        return result and on_boundary
# mark boundaries
gamma_inner = InnerCircle()
gamma_inner.mark(facet_marker, 1)
gamma_outer = OuterCircle()
gamma_outer.mark(facet_marker, 2)
#==============================================================================
# element and function space definition
cell = mesh.ufl_cell()
# taylor-hood element
elemV = dlfn.VectorElement("CG", cell, 2)
elemP = dlfn.FiniteElement("CG", cell, 1)
elemT = dlfn.FiniteElement("CG", cell, 1)
mixedElem = dlfn.MixedElement([elemV, elemP, elemT])
Wh = dlfn.FunctionSpace(mesh, mixedElem)
ndofs = Wh.dim()
ndofs_velocity = Wh.sub(0).dim()
ndofs_pressure = Wh.sub(1).dim()
ndofs_temperature = Wh.sub(2).dim()
print "DOFs velocity : ", ndofs_velocity, "\n" \
        "DOFs pressure : ", ndofs_pressure, "\n" \
        "DOFs temperature : ", ndofs_temperature 
#==============================================================================
# boundary conditions
if space_dim == 2:
    null_vector = dlfn.Constant((0.0, 0.0))
else:
    null_vector = dlfn.Constant((0.0, 0.0, 0.0))
inner_temperature = dlfn.Constant(0.5)
outer_temperature = dlfn.Constant(-0.5)
# initialize empty list
bcs = []
# no slip bc on all boundaries
bcs.append(dlfn.DirichletBC(Wh.sub(0), null_vector, facet_marker, 1))
bcs.append(dlfn.DirichletBC(Wh.sub(0), null_vector, facet_marker, 2))
# temperature bcs on left and right boundary
bcs.append(dlfn.DirichletBC(Wh.sub(2), inner_temperature, facet_marker, 1))
bcs.append(dlfn.DirichletBC(Wh.sub(2), outer_temperature, facet_marker, 2))
#==============================================================================
# definition of volume / surface element
dA = dlfn.Measure("ds", domain = mesh, subdomain_data = facet_marker)
dV = dlfn.Measure("dx", domain = mesh)
n = dlfn.FacetNormal(mesh)
# volume of domain (required for rms values)
V = dlfn.assemble(dlfn.Constant(1.0) * dV )
#==============================================================================
# trial and test function
(del_v, del_p, del_T) = dlfn.TestFunctions(Wh)
(dv, dp, dT) = dlfn.TrialFunctions(Wh)
#==============================================================================
# solution functions
sol = dlfn.Function(Wh)
sol0 = dlfn.Function(Wh)
sol00 = dlfn.Function(Wh)
v0, _, T0 = dlfn.split(sol0)
v00, _, T00 = dlfn.split(sol00)
#==============================================================================
# zero initial condition for velocity / pressure
if non_zero_initial_temperature:
    interp_initial_velocity = dlfn.interpolate(null_vector, Wh.sub(0).collapse())
    interp_initial_pressure = dlfn.interpolate(dlfn.Constant(0.), Wh.sub(1).collapse())
    if space_dim == 2:    
        radius_string = "sqrt( pow(x[0],2) + pow(x[1],2))"
    else:
        radius_string = "sqrt( pow(x[0],2) + pow(x[1],2) + pow(x[2],2))"
    if rotation == True:
        amplitude = 0.1
        x_string = "2 * {0} - ri - ro".format(radius_string)
        phi_string = "atan2(x[1], x[0])"
        # initial condition for temperature
        initial_temperature = dlfn.Expression(
            "ro*ri/{0} - ri + 210.0 * A / sqrt(17920.0 * pi) * \
                (1.0 - 3.0 * pow({1},2) + 3.0 * pow({1},2) - pow({1},6)) *\
                cos(4.0 * {2})".format(radius_string, x_string, phi_string),
            ri = inner_radius, ro = outer_radius, A = amplitude, pi = np.pi,
            element = elemT)

    else:
        # initial condition for temperature
        initial_temperature = dlfn.Expression(
            "(To - Ti) / (ro - ri) * (sqrt( pow(x[0],2) + pow(x[1],2)) -ri) + Ti ",
            ri = inner_radius, ro = outer_radius, Ti = inner_temperature, To = outer_temperature,
            element = elemT)
    interp_initial_temperature = dlfn.interpolate(initial_temperature, Wh.sub(2).collapse())
    # assign initial condition
    dlfn.assign(sol0, [interp_initial_velocity,
                       interp_initial_pressure,
                       interp_initial_temperature])
#==============================================================================
# define auxiliary operators
def a_operator(phi, psi):
    return inner(grad(phi), grad(psi)) * dV
def b_operator(phi, psi):
    return div(phi) * psi * dV
def c_operator(phi, chi, psi):
    return dot(dot(grad(chi), phi), psi) * dV
# 2D cross product
def cross_2D(phi):
    return dlfn.as_vector([phi[1], -phi[0]])
#==============================================================================
def print_imex_coefficients(a, b, c, omega, dt_ratio, dV):
    assert isinstance(a, list)
    assert isinstance(b, list)
    assert isinstance(c, list)
    assert isinstance(omega, dlfn.Constant)
    assert isinstance(dV, dlfn.Measure)
    assert isinstance(dt_ratio, float) and dt > 0.0
    V = dlfn.assemble(dlfn.Constant(1.) * dV)
    print "   time step ratio " , dt_ratio, " ",\
            "omega: ", dlfn.assemble(omega * dV) / V
    print "\n   a : ",
    for val in a:
        if val is not None:
            print dlfn.assemble(val * dV) / V, " ",
        else:
            print "None ",
    print "\n   b : ",
    for val in b:
        if val is not None:
            print dlfn.assemble(val * dV) / V, " ",
        else:
            print "None ", 
    print "\n   c : ",
    for val in c:
        if val is not None:
            print dlfn.assemble(val * dV) / V, " ",
        else:
            print "None ", 
    print "\n"
#==============================================================================
def compute_cfl_number(mesh, velocity, time_step):
    assert(isinstance(mesh, dlfn.Mesh))
    assert(isinstance(velocity, dlfn.Function))
    assert(isinstance(time_step, float) and time_step > 0.0)
    Vh_DG = dlfn.FunctionSpace(mesh, "DG", 0)
    h_cell = dlfn.CellDiameter(mesh)
    cfl_cell_expression = dlfn.sqrt(dlfn.inner(velocity, velocity)) * time_step / h_cell
    local_cfl = dlfn.project(cfl_cell_expression, Vh_DG)
    return local_cfl.vector().max()
#==============================================================================
def adjust_time_step(cfl, cfl_interval, time_steps, time_step_limits):
    # input processing
    assert(isinstance(cfl, float)) and cfl > 0.0
    assert(isinstance(time_steps, tuple) 
            and all(isinstance(x, float) for x in cfl_interval))
    assert(isinstance(cfl_interval, tuple) 
            and all(isinstance(x, float) for x in cfl_interval))
    assert(isinstance(time_step_limits, tuple) 
            and all(isinstance(x, float) for x in time_step_limits))
    cfl_min, cfl_max = cfl_interval
    assert cfl_min < cfl_max and cfl_min > 0 and cfl_max > 0
    dt_min, dt_max = time_step_limits
    assert dt_min < dt_max and dt_min > 0 and dt_max > 0
    old_time_step, old_old_time_step = time_steps
    assert old_time_step > 0 and old_time_step > 0
    # set boolean flags
    update_coefficients = False
    # compute intended time step
    dt_cfl = 0.5 * (cfl_min + cfl_max) / cfl * old_time_step
    # initial new dummy time step
    new_time_step = -1.0
    # check time step should be modified
    if (cfl > cfl_max or cfl < cfl_min) and dt_cfl < dt_max:
        if dt_cfl < dt_min:
            raise ValueError("time step gets too small, aborting this run")
        elif dt_cfl != old_time_step:
            new_time_step = dt_cfl
            update_coefficients = True
    elif dt_cfl > dt_max and old_time_step != dt_max:
        new_time_step = dt_max
        update_coefficients = True
    elif (dt_cfl > dt_max and old_time_step == dt_max) or \
            (cfl < cfl_max and cfl > cfl_min):
        new_time_step = old_time_step
    if update_coefficients:
        print "   Modified time step from " + \
                "{0:03.2e} to {1:03.2e}".format(old_time_step, new_time_step)
    elif new_time_step != old_old_time_step:
        update_coefficients = True
    assert new_time_step > 0.
    return update_coefficients, new_time_step
#==============================================================================
# constant gravity in radial direction
if space_dim == 2:
    gravity_vector = dlfn.Expression(
                    ('x[0] / sqrt( pow(x[0],2) + pow(x[1],2) )', 
                     'x[1] / sqrt( pow(x[0],2) + pow(x[1],2) )'), degree=2)
elif space_dim == 3:
    gravity_vector = dlfn.Expression(
                    ('x[0] / sqrt( pow(x[0],2) + pow(x[1],2) + pow(x[2],2) )', 
                     'x[1] / sqrt( pow(x[0],2) + pow(x[1],2) + pow(x[2],2) )',
                     'x[2] / sqrt( pow(x[0],2) + pow(x[1],2) + pow(x[2],2) )'),
                    degree=2)
#==============================================================================
# coefficients for non-equidistant time stepping
# time step ratio
omega = dlfn.Constant(1.0)
# some constants...
one = dlfn.Constant(1.0)
two = dlfn.Constant(2.0)
# coefficients of time derivative
a = [(one + two * omega) / (one + omega),
     (one + omega),
     dlfn.sqr(omega)/(one + omega)]
# coefficients of nonlinear term
b = [None,
     (one + omega),
     omega]
# coefficients of linear term
c = [one, None , None]
#==============================================================================
# defining extrapolated quantities
extrapolated_temperature = (one + omega) * T0 - omega * T00
extrapolated_velocity = (one + omega) * v0 - omega * v00
#==============================================================================
# setting initial time step
time_step = dlfn.Constant(dt_initial)
dt = dt_initial
dt_old = dt_initial
#===============================================================================
# define linear variational problem
# momentum equation
lhs_momentum = a[0] * dot(dv, del_v) *  dV \
             + time_step * c[0] * equation_coefficients[1] * a_operator(dv, del_v) \
             - time_step * b_operator(del_v, dp) \
             - time_step * b_operator(dv, del_p)
rhs_momentum = a[1] * dot(v0, del_v) * dV \
             - a[2] * dot(v00, del_v) * dV \
             - time_step * b[1] * c_operator(v0, v0, del_v) \
             + time_step * b[2] * c_operator(v00, v00, del_v) \
             + time_step * equation_coefficients[2] * extrapolated_temperature * dot(gravity_vector, del_v) * dV
if rotation == True:
    print "\nAdding rotation to the model...\n"
    if space_dim == 2:
        rhs_momentum += - time_step * equation_coefficients[0] * dot(cross_2D(extrapolated_velocity), del_v) * dV
    elif space_dim == 3:
        rotation_vector = dlfn.Constant((0., 0., 1.))
        rhs_momentum += - time_step * equation_coefficients[0] * dot(
                            dlfn.cross(rotation_vector, extrapolated_velocity),
                            del_v) * dV
# energy equation
lhs_energy = a[0] * dot(dT, del_T) * dV \
           + time_step * equation_coefficients[3] * a_operator(dT, del_T)
rhs_energy = a[1] * dot(T0, del_T) * dV \
           - a[2] * dot(T00, del_T) * dV \
           - time_step * b[1] * c_operator(v0, T0, del_T) \
           + time_step * b[2] * c_operator(v00, T00, del_T)
lhs = lhs_momentum + lhs_energy
#===============================================================================
# full problem
lhs = lhs_momentum + lhs_energy
rhs = rhs_momentum + rhs_energy
if not use_assembler_method:
    # linear problem
    problem = dlfn.LinearVariationalProblem(lhs, rhs, sol, bcs=bcs)
    solver = dlfn.LinearVariationalSolver(problem)
else:
    # system assembler
    assembler = dlfn.SystemAssembler(lhs, rhs, bcs=bcs)
    system_matrix = dlfn.Matrix()
    system_rhs = dlfn.Vector()
    solver = dlfn.LUSolver(system_matrix)
#===============================================================================
def solve(step, time_step_ratio, update_coefficients, use_assembler_method):
    assert isinstance(step, int) and step >= 0
    assert isinstance(time_step_ratio, float) and time_step_ratio > 0
    assert isinstance(update_coefficients, bool) \
        and isinstance(use_assembler_method, bool)
    if step == 0:
        # assign omega for initial Euler step
        omega.assign(0.)
        assert isinstance(omega, dlfn.Constant)
        # assemble system matrix / rhs
        if use_assembler_method:
            print "   Assembling system (initial step)..."
            assembler.assemble(system_matrix, system_rhs)
            # re-compute LU decomposition
            solver.parameters['reuse_factorization'] = False
        update_coefficients = False
    elif update_coefficients:
        # assign omega for SBF2 step
        omega.assign(time_step_ratio)
        assert isinstance(omega, dlfn.Constant)
        # assemble system matrix / rhs
        if use_assembler_method:
            print "   Assembling system (initial step)..."
            assembler.assemble(system_matrix, system_rhs)
            # re-compute LU decomposition
            solver.parameters['reuse_factorization'] = False
        update_coefficients = False
    elif use_assembler_method:
        print "   Assembling rhs..."
        assembler.assemble(system_rhs)
    #===========================================================================
    if (print_time_stepping_coefficients):
        print_imex_coefficients(a, b, c, omega, time_step_ratio, dV)
    #===========================================================================
    # solve linear problem
    if use_assembler_method:
        solver.solve(sol.vector(), system_rhs)
        solver.parameters['reuse_factorization'] = True
    else:
        solver.solve()
#==============================================================================
# initialization
update_coefficients = False
step = 0
checkpoint_cnt = 0
rms_cnt = 0
output_cnt = 0
time = 0.0
# array for time logging (time, cfl number, time step)
time_logging = np.zeros((n_steps, 3), dtype = np.float)
# array for checkpoint logging of (step number, time)
checkpoint_logging = np.zeros((n_steps / checkpoint_frequency, 2), dtype=object)
checkpoint_logging[:,0] = 0
checkpoint_logging[:,1] = 0.0
# array for output logging of (step number, time)
output_logging = np.zeros((n_steps / output_frequency, 2), dtype=object)
output_logging[:,0] = 0
output_logging[:,1] = 0.0
# array for rms logging (step number, time, velocity_rms, temperature_rms)
rms_logging = np.zeros((n_steps / rms_frequency, 4), dtype=object)
rms_logging[:,0] = 0
rms_logging[:,1] = 0.0
# dolfin timer 
dlfn.tic()
#==============================================================================
# time loop
while time < t_end and step < n_steps:
    print "Iteration: {:08d}, ".format(step), "time = {0:10.5f},".format(time),\
            " time step = {0:5.4e}".format(dt)
    #===========================================================================
    # compute solution
    solve(step, dt / dt_old, update_coefficients, use_assembler_method)
    # extract solution
    velocity, pressure, temperature = sol.split()
    #===========================================================================
    # compute cfl number
    cfl = compute_cfl_number(mesh, velocity, dt)
    print "   Current cfl number: {0:03.2e}".format(cfl)
    # reject step if clf number > 1.0
    if cfl > 1.0:
        print "   Time step rejected".format(cfl)
        # decrease time step
        dt /= 10.0
        # recompute time step
        continue
    #===========================================================================
    # time step is accepted
    time_logging[step] = time, cfl, dt
    #===========================================================================
    # modify time step
    if step != 0 and adaptive_time_stepping:
        dt_old_old = dt_old
        dt_old = dt
        update_coefficients, dt = adjust_time_step(cfl,
                                                   (cfl_min, cfl_max),
                                                   (dt_old, dt_old_old),
                                                   (dt_min, dt_max))
        time_step.assign(dt)
    elif step == 0:
        # update coefficients after initial step to switch to SBF2
        dt_old = dt
        update_coefficients = True
    #===========================================================================
    # write output
    if step % output_frequency == 0:
        print "   Writing output..."
        # normalize pressure to mean value pressure
        mean_pressure = dlfn.assemble(pressure * dV) / V
        normalized_pressure = dlfn.project(pressure - mean_pressure,
                                           dlfn.FunctionSpace(mesh, elemP))
        # write output to files
        pvd_velocity << (velocity, time)
        pvd_pressure << (pressure, time)
        pvd_temperature << (temperature, time)
        # write log
        output_logging[output_cnt] = step, time
        output_cnt += 1
    #===========================================================================
    # compute rms values
    if step % rms_frequency == 0:
        print "   Computing rms-values..."
        velocity_rms = np.sqrt(dlfn.assemble(dot(velocity, velocity) * dV) / V)
        temperature_rms = np.sqrt(dlfn.assemble(dlfn.sqr(temperature) * dV) / V)
        print "   rms velocity: {0:3.2e}, rms temperature: {1:3.2e}".format(
                velocity_rms, temperature_rms)
        if velocity_rms > 1e6 or temperature_rms > 0.5:
            raise ValueError("Instability occured!")
        # write log
        rms_logging[rms_cnt] = step, time, velocity_rms, temperature_rms
        rms_cnt += 1
    #===========================================================================
    # write checkpoint
    if step % checkpoint_frequency == 0 and step != 0:
        print "   Writing checkpoint..."
        # write current solution for restart
        fname = "checkpoints/" + "solution-{:06d}.h5".format(step)
        output_file = dlfn.HDF5File(mesh.mpi_comm(), fname, "w")
        output_file.write(sol, "solution")
        output_file.close()
        # write old solution for restart
        fname = "checkpoints/" + "old_solution-{:06d}.h5".format(step)
        output_file = dlfn.HDF5File(mesh.mpi_comm(), fname, "w")
        output_file.write(sol0, "old_solution")
        output_file.close()
        # save checkpoint time
        checkpoint_logging[checkpoint_cnt] = step, time
        checkpoint_cnt += 1
    #===========================================================================
    # update solutions for next iteration
    sol00.assign(sol0)
    sol0.assign(sol)
    time += dt
    #===========================================================================
    # increase counter
    step += 1
    #===========================================================================
    # asserts to guarantee that variable type are correct
    assert isinstance(time_step, dlfn.Constant)
    assert isinstance(dt, float) and isinstance(dt_old, float)
print "elapsed simulation time: ", dlfn.toc(), " sec"
#===========================================================================
print "\nCreating plots... "
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].semilogy(time_logging[:,2])
ax[0].set_ylabel("dt")
ax[1].plot(time_logging[:,1])
ax[1].set_xlabel("step number")
ax[1].set_ylabel("cfl")
plt.savefig("time_step_history.pdf")
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(rms_logging[:,1], rms_logging[:,2])
ax[0].set_ylabel("velocity")
ax[1].plot(rms_logging[:,1], rms_logging[:,3])
ax[1].set_xlabel("time")
ax[1].set_ylabel("temperature")
plt.savefig("rms_history.pdf")
#===========================================================================
# resize time arrays
time_logging = time_logging[:(step - 1)]
assert time_logging[-1,0] != 0
if checkpoint_cnt > 0:
    checkpoint_logging = checkpoint_logging[:checkpoint_cnt]
    assert checkpoint_logging[-1,0] != 0
if rms_cnt > 0:
    rms_logging = rms_logging[:rms_cnt]
    assert rms_logging[-1,0] != 0
if output_cnt > 0:
    output_logging = output_logging[:rms_cnt]
    assert rms_logging[-1,0] != 0
# write csv files

for fname, data in [("output_log.csv", output_logging),
                    ("rms_log.csv", rms_logging),
                    ("checkpoint_log.csv", checkpoint_logging)]:
    n_cols = data.shape[1]
    fmt = tuple(['%d'] + (n_cols - 1) * ['%.6e', ])
    np.savetxt(fname, data, fmt=fmt)
    print "Written record to " + fname