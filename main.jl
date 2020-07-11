using StaticArrays: StaticArrays
using Rotations
using CoordinateTransformations

include("GeometryDisplay.jl")
include("SPLFieldModel.jl")
using .GeometryDisplay
using .SPLFieldModel

spl_field = SPLField()
visualize(spl_field.lines)


