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

function distance_segment(point, line; p = 0.2)
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

    dp = point - closest_point_on_line
    norm(dp, p)
end

"The line fit error of a single `line` - `map_line` pair."
@inline function line_fit_error(line::Line, map_line::Line, p)
    # technically, the right thing would be to comupte the sum of all distances between the
    # `lines` and the respective closest field `map_lines`. Thus, this would boild down
    # to the absolute area bewetten the perceived lines and their respective closeset field line.
    #
    # for simplicty, let's just use the distnace of the end-points of the perceived lines to a field
    # line.
    # TODO: Read about a better error metric here (ICL paper)
    d1 = distance_segment(line[1], map_line; p)
    d2 = distance_segment(line[2], map_line; p)
    d1 + d2
end

"The line fit error if a single line to the entire map of `map_lines`."
# TODO: Use a more robust association logic here:
#   - only consider maplines that are long enough
#   - explictly filter for orientation
#   - only add robust kernels
@inline function line_fit_error(line::Line, map_lines::AbstractVector, p)
    line_len_sq = length_sq(line)
    map_line_candidates = filter(l -> length_sq(l) > line_len_sq, map_lines)
    @assert !isempty(map_line_candidates)
    minimum(ml -> line_fit_error(line, map_line, p), map_line_candidates) * line_len_sq
end

function fit_transformation(
    lines,
    map_lines;
    n_iterations_max = 50,
    snapshot_stepsize = 1,
    min_grad_norm = 1e-3,
    min_cost = 0.1,
    optimizer = Optimizers.LevenBergMarquardt(),
    p = 0.2,
)

    θ::PoseTransformation{Float64} = zero(PoseTransformation)
    "Normalize transformation to rotate around com of lines."
    rot_center, _ = center_of_mass(lines)
    debug_snapshots = []

    optimizer_state = Optimizers.initial_state(optimizer)
    grad_result = DiffResults.GradientResult(θ)

    function cost(params)
        tform = pose_transformation(params; rot_center)
        c = sum(l -> line_fit_error(tform(l), map_lines, p), lines)
    end

    converged = false
    cost_decreased = true

    for i in 1:n_iterations_max
        println("outer_i: $i")
        grad_result = ForwardDiff.gradient!(grad_result, cost, θ)
        ∇θ = DiffResults.gradient(grad_result)
        V = DiffResults.value(grad_result)
        println("$grad_result")
        if V < min_cost
            converged = true
            break
        end
        if !cost_decreased
            break
        end

        θ, optimizer_state, cost_decreased =
            Optimizers.step(optimizer, cost, V, θ, ∇θ, optimizer_state)

        # take a snapshot every few iterations
        if !isnothing(snapshot_stepsize) && iszero(i % snapshot_stepsize)
            tform_snapshot = pose_transformation(θ; rot_center)
            cost_snapshot = cost(θ)
            push!(debug_snapshots, (; i, tform_snapshot, cost_snapshot))
        end
    end

    converged ? @info("Converged!") : @warn("Not converged!")

    pose_transformation(θ; rot_center), converged, debug_snapshots
end

end # module
