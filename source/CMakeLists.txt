# Set the name of the project and target:
SET(TARGET "main")

SET(TARGET_SRC
  main.cc
  grid_factory.cc
  timestepping.cc
  conducting_fluid/assemble_system.cc
  conducting_fluid/conducting_fluid_solver.cc
  conducting_fluid/make_grid.cc
  conducting_fluid/setup.cc
  # You can specify additional files here!
)


PROJECT(${TARGET} CXX)

INCLUDE_DIRECTORIES(
  include
  ${CMAKE_SOURCE_DIR}/include
  )

DEAL_II_INVOKE_AUTOPILOT()
