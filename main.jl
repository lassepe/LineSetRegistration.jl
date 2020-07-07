using GeometryBasics: Line, Point, decompose

using StaticArrays: StaticArrays
using Rotations
using CoordinateTransformations

include("GeometryDisplay.jl")
include("SPLFieldModel.jl")


GeometryDisplay.visualize(field_lines)
