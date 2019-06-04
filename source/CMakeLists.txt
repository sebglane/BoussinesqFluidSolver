# Set the name of the project and target:
SET(TARGET "main")

SET(TARGET_SRC
  main.cc
  grid_factory.cc
  timestepping.cc
  assemble_system.cc
  magnetic_diffusion_solver.cc
  initial_values.cc
  make_grid.cc
  setup.cc
  solve_magnetic.cc
  # You can specify additional files here!
)


PROJECT(${TARGET} CXX)

INCLUDE_DIRECTORIES(
  include
  ${CMAKE_SOURCE_DIR}/include
  )

DEAL_II_INVOKE_AUTOPILOT()
