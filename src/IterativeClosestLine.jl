module IterativeClosestLine

using CoordinateTransformations: CoordinateTransformations, LinearMap
using ForwardDiff
using GeometryBasics: Point, Line
using LinearAlgebra: ⋅, I, norm

# include("GeometryTransformationUtils.jl")
using ..GeometryTransformationUtils:
    center_of_mass, PoseTransformation, pose_transformation, direction_vector, length_sq

# include("Optimizers.jl")
import ..Optimizers

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
    line_filter_tolerance = 1e-3,
)
    observed_line_len_sq = length_sq(observed_line)
    map_line_candidates =
        filter(ml -> (length_sq(ml) + line_filter_tolerance >= observed_line_len_sq), map_lines)
    @assert !isempty(map_line_candidates)
    minimum(ml -> line_fit_error(observed_line, ml, p), map_line_candidates) * sqrt(observed_line_len_sq)
end

function fit_transformation(
    observed_lines,
    map_lines;
    n_iterations_max = 50,
    min_grad_norm = 1e-3,
    min_cost = 0.1,
    optimizer = Optimizers.LevenBergMarquardt(),
    p = 1.1,
)

    θ::PoseTransformation{Float64} = zero(PoseTransformation)
    "Normalize transformation to rotate around com of lines."
    rotation_center, _ = center_of_mass(observed_lines)

    optimizer_state = Optimizers.initial_state(optimizer)
    grad_result = DiffResults.GradientResult(θ)

    function cost((x, y, α))
        tform = pose_transformation(x, y, α; rotation_center)
        c = sum(l -> line_fit_error(tform(l), map_lines, p), observed_lines)
    end

    converged = false
    cost_decreased = true

    for i in 1:n_iterations_max
        grad_result = ForwardDiff.gradient!(grad_result, cost, θ)
        ∇θ = DiffResults.gradient(grad_result)
        V = DiffResults.value(grad_result)
        println(∇θ)
        if V < min_cost
            converged = true
            break
        end
        if !cost_decreased
            break
        end

        θ, optimizer_state, cost_decreased =
            Optimizers.step(optimizer, cost, V, θ, ∇θ, optimizer_state)
    end

    pose_transformation(θ...; rotation_center), converged
end

end # module
