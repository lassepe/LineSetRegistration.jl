using CoordinateTransformations: CoordinateTransformations, Translation, LinearMap
using DataFrames: DataFrame, nrow
using ForwardDiff
using GeometryBasics: Point, Line
using LinearAlgebra: ⋅, I, diagm, norm
using MacroTools: @forward
using Random: Random, shuffle
using StaticArrays: FieldVector, SMatrix, SVector, SizedVector, SDiagonal
using VegaLite

using RCall
@rlibrary ggplot2
@rlibrary gganimate

show_debug_animation = true

# TODO: dirty fix for "invalid redifition of constant" bug in julia master.
if !@isdefined INCLUDED
    @eval begin
        INCLUDED = true
        include("spl_field_model.jl")
    end
end

include("visualize.jl")
include("problem_setup.jl")
include("transformation_utils.jl")
include("optimizer.jl")
include("fit_line_transformation.jl")
include("test_data.jl")

#====================================== optimization problem ======================================#
initial_lines, static_line_data, debug_snapshots = run_test()
