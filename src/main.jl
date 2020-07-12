using CoordinateTransformations: Transformation, LinearMap
using DataFrames
using GeometryBasics: Point, Line
using Lazy: @forward
using LinearAlgebra: diagm, norm
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
perceived_lines = [Line(rand(Point{2}), rand(Point{2})) for _ in 1:2]

visualize(vcat(
    line_dataframe(spl_field.lines, "map"),
    line_dataframe(perceived_lines, "perceived"),
)) |> display

#====================================== optimization problem ======================================#


function line_fit_error(perceived_lines, map_lines)
    # technically, the right thing would be to comupte the sum of all distances between the
    # `perceived_lines` and the respective closest field `map_lines`. Thus, this would boild down
    # to the absolute area bewetten the perceived lines and their respective closeset field line.

    # for simplicty, let's just use the distnace of the end-points of the perceived lines to a field
    # line.

    function distance(point::Point, line::Line)
        (x0, y0) = point
        (x1, y1), (x2, y2) = p1, p2 = line
        abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / norm(p1 - p2)
    end

    function line_fit_error(line1::Line, line2::Line)
        d1 = distance(line1[1], line2)
        d2 = distance(line1[2], line2)
        d1 + d2
    end

    sum(perceived_lines) do perceived_line
        minimum(map_lines) do field_line
            line_fit_error(perceived_line, field_line)
        end
    end
end