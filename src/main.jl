using CoordinateTransformations: Transformation, LinearMap
using DataFrames
using GeometryBasics: Point, Line
using Lazy: @forward
using LinearAlgebra: diagm
using RCall
using Rotations
using StaticArrays: SMatrix, StaticArrays
using VegaLite

@rlibrary ggplot2

include("visualize.jl")
include("spl_field_model.jl")

"The known lines on the map."
spl_field = SPLField()
"Some perceived lines."
perceived_lines = [Line(rand(Point{2}), rand(Point{2}))]
""

visualize(vcat(line_dataframe(spl_field.lines, "map"), line_dataframe(perceived_lines, "perceived")))
