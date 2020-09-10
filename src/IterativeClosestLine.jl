module IterativeClosestLine

using CoordinateTransformations: CoordinateTransformations, LinearMap
using ForwardDiff
using GeometryBasics: Point, Line
using LinearAlgebra: ⋅, I, diagm, norm
using StaticArrays: FieldVector, SMatrix, SVector, SizedVector, SDiagonal

struct PoseTransformation{T} <: FieldVector{3,T}
    x::T
    y::T
    α::T
end

function line_vector(line::Line)
    p1, p2 = line
    p2 - p1, p1, p2
end

function line_length_sq(line::Line)
    line_vec, _ = line_vector(line)
    line_vec ⋅ line_vec
end

function distance_segment_sq(point, line)
    line_vec, p_start, p_end = line_vector(line)
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
    dp ⋅ dp
end

function distance_segment(point, line)
    sqrt(distance_segment_sq(point, line))
end

"The line fit error of a single `line` - `map_line` pair."
@inline function line_fit_error(line::Line, map_line::Line)
    # technically, the right thing would be to comupte the sum of all distances between the
    # `lines` and the respective closest field `map_lines`. Thus, this would boild down
    # to the absolute area bewetten the perceived lines and their respective closeset field line.
    #
    # for simplicty, let's just use the distnace of the end-points of the perceived lines to a field
    # line.
    d1, d2 = distance_segment(line[1], map_line), distance_segment(line[2], map_line)
    sqrt(line_length_sq(line)) * (d1 + d2)
end

"The line fit error if a single line to the entire map of `map_lines`."
@inline function line_fit_error(line::Line, map_lines::AbstractVector)
    minimum(map_lines) do map_line
        line_fit_error(line, map_line)
    end
end

function fit_transformation(
    lines,
    map_lines;
    n_iterations_max = 50,
    snapshot_stepsize = false,
    min_grad_norm = 1e-3,
    min_cost = 0.1,
    optimizer = LevenBergMarquardt(),
)

    θ::PoseTransformation{Float64} = zero(PoseTransformation)
    "Normalize transformation to rotate around com of lines."
    lines_com, lines_mass = center_of_mass(lines)
    debug_snapshots = []

    optimizer_state = initial_state(optimizer)
    grad_result = DiffResults.GradientResult(θ)

    function cost(params)
        tform = pose_transformation(params; rot_center = lines_com)
        c = sum(l -> line_fit_error(tform(l), map_lines), lines)
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
            optimizer_step(optimizer, cost, V, θ, ∇θ, optimizer_state)

        # take a snapshot every few iterations
        if !isnothing(snapshot_stepsize) && iszero(i % snapshot_stepsize)
            tform_snapshot = pose_transformation(θ; rot_center = lines_com)
            cost_snapshot = cost(θ)
            push!(debug_snapshots, (; i, tform_snapshot, cost_snapshot))
        end
    end

    converged ? @info("Converged!") : @warn("Not converged!")

    pose_transformation(θ; rot_center = lines_com), converged, debug_snapshots
end

end # module
