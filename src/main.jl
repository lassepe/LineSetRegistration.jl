using CoordinateTransformations: Translation, LinearMap
using DataFrames: DataFrame
using GeometryBasics: Point, Line
using Lazy: @forward
using LinearAlgebra: diagm, norm
using RCall
using StaticArrays: SMatrix, StaticArrays
using VegaLite

@rlibrary ggplot2

include("visualize.jl")
include("spl_field_model.jl")

function transform(tform, line)
    Line(tform.(Tuple(line))...)
end

"The known lines on the map."
spl_field = SPLField()
"Some perceived lines."
perceived_lines = [Line(Point(1, 0), Point(2, 0)), Line(Point(2, 0), Point(2, 2))]

#======================================= optimization utils =======================================#

function line_fit_error(perceived_lines, map_lines)
    # technically, the right thing would be to comupte the sum of all distances between the
    # `perceived_lines` and the respective closest field `map_lines`. Thus, this would boild down
    # to the absolute area bewetten the perceived lines and their respective closeset field line.
    #
    # for simplicty, let's just use the distnace of the end-points of the perceived lines to a field
    # line.

    function distance(point::Point, line::Line)
        (x0, y0) = point
        (x1, y1), (x2, y2) = p1, p2 = line
        abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / norm(p1 - p2)
    end

    function line_length(line::Line)
        p1, p2 = line
        norm(p1 - p2)
    end

    function line_fit_error(perceived_line::Line, map_line::Line)
        d1, d2 = distance(perceived_line[1], map_line), distance(perceived_line[2], map_line)
        line_length(perceived_line) * (d1 + d2) / 2
    end

    sum(perceived_lines) do perceived_line
        minimum(map_lines) do field_line
            line_fit_error(perceived_line, field_line)
        end
    end
end

function pose_transformation(Δx, Δy, Δα)
    sα, cα = sin(Δα), cos(Δα)
    Translation(Δx, Δy) ∘ LinearMap(SMatrix{2,2}(cα, sα, -sα, cα))
end

transformed_lines = map(l -> transform(pose_transformation(1, 1, pi/4), l), perceived_lines)

visualize(vcat(
    line_dataframe(spl_field.lines, "map"),
    line_dataframe(perceived_lines, "perceived"),
    line_dataframe(transformed_lines, "transformed"),
)) |> display
