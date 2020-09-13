module IterativeClosestLine

using CoordinateTransformations: CoordinateTransformations, LinearMap
using ForwardDiff
using GeometryBasics: Point, Line
using LinearAlgebra: ⋅, I, norm

using ..GeometryTransformationUtils:
    center_of_mass, PoseTransformation, pose_transformation, direction_vector, length_sq

import Optim

function distance_segment(point, line, p)
    line_vec, p_start, p_end = direction_vector(line)
    line_len_sq = line_vec ⋅ line_vec
    @assert line_len_sq > 1e-3
    t = ((point - p_start) ⋅ line_vec) / line_len_sq

    closest_point_on_line = if t < 0
        p_start
    elseif t > 1
        p_end
    else
        p_start + line_vec .* t
    end

    norm(point - closest_point_on_line, p)
end

"The line fit error of a single `line` - `map_line` pair."
@inline function line_fit_error(observed_line::Line, map_line::Line, p)
    # technically, the right thing would be to comupte the sum of all distances between the
    # `observed_line` and the respective closest field `map_lines`. Thus, this would boild down
    # to the absolute area bewetten the perceived lines and their respective closeset field line.
    #
    # for simplicty, let's just use the distnace of the end-points of the perceived lines to a field
    # line.
    # TODO: Read about a better error metric here (ICL paper)
    d1 = distance_segment(observed_line[1], map_line, p)
    d2 = distance_segment(observed_line[2], map_line, p)
    d1 + d2
end

"The line fit error if a single line to the entire map of `map_lines`."
# TODO: Use a more robust association logic here:
#   - only consider maplines that are long enough
#   - explictly filter for orientation
#   - only add robust kernels
@inline function line_fit_error(
    observed_line::Line,
    map_lines::AbstractVector{<:Line},
    p;
    length_filter_tolerance = 1e-1,
)
    observed_line_len_sq = length_sq(observed_line)
    map_line_candidates =
        filter(ml -> (length_sq(ml) + length_filter_tolerance >= observed_line_len_sq), map_lines)
    @assert !isempty(map_line_candidates)
    minimum(ml -> line_fit_error(observed_line, ml, p), map_line_candidates) *
    sqrt(observed_line_len_sq)
end

function fit_transformation(
    observed_lines,
    map_lines;
    min_grad_norm = 1e-3,
    min_cost = 0.1,
    p = 1.1,
)
    # Normalize transformation to rotate around center of mass
    rotation_center, _ = center_of_mass(observed_lines)

    function cost((x, y, α))
        tform = pose_transformation(x, y, α; rotation_center)
        sum(l -> line_fit_error(tform(l), map_lines, p), observed_lines)
    end

    @time res = Optim.optimize(
        cost,
        zeros(3),
        Optim.LBFGS(),
        Optim.Options(iterations = 200);
        autodiff = :forward,
    )
    θ = Optim.minimizer(res)
    @show Optim.iterations(res)
    @show converged = Optim.converged(res)
    @show minimum = Optim.minimum(res)

    pose_transformation(θ...; rotation_center), converged
end

end # module
