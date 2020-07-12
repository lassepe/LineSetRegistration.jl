using CoordinateTransformations: Translation, LinearMap
using DataFrames: DataFrame
using GeometryBasics: Point, Line
using Lazy: @forward
using LinearAlgebra: diagm, norm
using RCall
using StaticArrays: SMatrix, SVector, SizedVector
using VegaLite
using Flux: Flux, ADAM
using ForwardDiff
using LinearAlgebra: ⋅

@rlibrary ggplot2

include("visualize.jl")
include("spl_field_model.jl")

function transform(tform, line)
    Line(tform.(Tuple(line))...)
end

"The known lines on the map."
spl_field = SPLField()
"Some perceived lines."
perceived_lines = [Line(Point(1., -2.), Point(2., -1.9)), Line(Point(1., -2), Point(0.8, 0.0))]

#======================================= optimization utils =======================================#

function line_fit_error(lines, map_lines)
    # technically, the right thing would be to comupte the sum of all distances between the
    # `lines` and the respective closest field `map_lines`. Thus, this would boild down
    # to the absolute area bewetten the perceived lines and their respective closeset field line.
    #
    # for simplicty, let's just use the distnace of the end-points of the perceived lines to a field
    # line.

    function distance_projected(point::Point, line::Line)
        (x0, y0) = point
        (x1, y1), (x2, y2) = p1, p2 = line
        abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / norm(p1 - p2)
    end

    function distance_segment(point::Point, line::Line)
        p_start, p_end = line
        line_len = norm(p_end - p_start)
        line_vector = p_end - p_start

        if line_len ≈ 0
            return norm(p_start - point)
        end

        t = ((point - p_start) ⋅ line_vector) / line_len

        closest_point_on_line = if t < 0
            p_start
        elseif t > 1
            p_end
        else
            p_start + line_vector .* t
        end
        norm(point - closest_point_on_line)
    end

    function line_length(line::Line)
        p1, p2 = line
        norm(p1 - p2)
    end

    function line_fit_error(line::Line, map_line::Line)
        d1, d2 = distance_segment(line[1], map_line), distance_segment(line[2], map_line)
        line_length(line) * (d1 + d2) / 2
    end

    sum(lines) do line
        minimum(map_lines) do map_line
            line_fit_error(line, map_line)
        end
    end
end

function pose_transformation(transformation_parameters)
    pose_transformation(
        transformation_parameters[1],
        transformation_parameters[2],
        transformation_parameters[3],
    )
end

function pose_transformation(Δx, Δy, Δα)
    sα, cα = sin(Δα), cos(Δα)
    Translation(Δx, Δy) ∘ LinearMap(SMatrix{2,2}(cα, sα, -sα, cα))
end

#====================================== optimization problem ======================================#

function fit_line_transformation(lines, map_lines; n_iterations_max = 100, optimizer = ADAM())
    "Initial guess of parameters we want to fit."
    transformation_parameters = zero(SizedVector{3, Float64})
    transformation_gradient = similar(transformation_parameters)

    function objective(params)
        tform = pose_transformation(params)
        transformed_lines = map(l -> transform(tform, l), lines)
        line_fit_error(transformed_lines, map_lines)
    end

    for i in 1:n_iterations_max
        # TODO: make a non-allocating version here.
        ForwardDiff.gradient!(transformation_gradient, objective, transformation_parameters)
        # TODO: make a non-allocating version here.
        Flux.update!(optimizer, transformation_parameters, transformation_gradient)
    end

    pose_transformation(transformation_parameters)
end

fitted_line_transformation = fit_line_transformation(perceived_lines, spl_field.lines)
transformed_lines = map(l -> transform(fitted_line_transformation, l), perceived_lines)

visualize(vcat(
    line_dataframe(spl_field.lines, "map"),
    line_dataframe(perceived_lines, "perceived"),
    line_dataframe(transformed_lines, "transformed"),
)) |> display
