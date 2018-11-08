# Set the name of the project and target:
SET(TARGET "main")

SET(TARGET_SRC
  main.cc
  assembly_data.cc
  buoyant_fluid_solver.cc
  initial_values.cc
  parameters.cc
  postprocessor.cc
  preconditioning.cc
  timestepping.cc
  # You can specify additional files here!
)

PROJECT(${TARGET} CXX)

INCLUDE_DIRECTORIES(
  include
  ${CMAKE_SOURCE_DIR}/include
  )

DEAL_II_INVOKE_AUTOPILOT()
