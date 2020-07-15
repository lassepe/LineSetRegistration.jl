using CoordinateTransformations: Translation, LinearMap
using DataFrames: DataFrame, nrow
using ForwardDiff
using GeometryBasics: Point, Line
using LinearAlgebra: I, diagm, norm
using LinearAlgebra: ⋅
using MacroTools: @forward
using RCall
using Random: Random
using StaticArrays: FieldVector, SMatrix, SVector, SizedVector, SDiagonal
using VegaLite

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

function transform(tform, line)
    Line(tform.(Tuple(line))...)
end

#======================================= optimization utils =======================================#

function line_vector_representation(line::Line)
    p1, p2 = line
    p2 - p1, p1, p2
end

function line_length_sq(line::Line)
    line_vec, _ = line_vector_representation(line)
    line_vec ⋅ line_vec
end

function distance_segment_sq(point, line)
    line_vec, p_start, p_end = line_vector_representation(line)
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

function pose_transformation(transformation_parameters; kwargs...)
    pose_transformation(
        transformation_parameters[1],
        transformation_parameters[2],
        transformation_parameters[3];
        kwargs...,
    )
end

function pose_transformation(x, y, α; rot_center = zero(Point{2}))
    sα, cα = sin(α), cos(α)
    Translation(rot_center) ∘ Translation(x, y) ∘ LinearMap(SMatrix{2,2}(cα, sα, -sα, cα)) ∘
    inv(Translation(rot_center))
end

#=================================== thin optimizer abstraction ===================================#

abstract type FirstOrderOptimizer end
function initial_state end
function first_order_step end

Base.@kwdef struct LevenBergMarquardt <: FirstOrderOptimizer
    "The scaling factor for adaptive damping."
    ρ::Float64 = 1.0
    "The initial regularization factor."
    λ₀::Float64 = 1.0
end

function initial_state(optimizer::LevenBergMarquardt)
    optimizer.λ₀
end

function first_order_step(optimizer::LevenBergMarquardt, Vfunc, V, θ, ∇θ, λ)
    # TODO: introduce line-search for dynamic damping.
    # levenberg-marquard step
    M = ∇θ * ∇θ'
    for i in 1:100
        θ_candidate = θ - (M + λ * I) \ ∇θ * V
        V_candidate = Vfunc(θ_candidate)
        println(V_candidate)
        println(i)
        println("=")
        # if the cost decreased, accept the step candidate
        if V_candidate < V
            # accept step candidate
            return (; step = θ_candidate, λ = λ / optimizer.ρ)
        end
        # the step was not accepted, adjust the damping
        λ *= optimizer.ρ
    end
    @assert false "Did not converge. $λ"
end

Base.@kwdef struct Descent <: FirstOrderOptimizer
    step_size::Float64 = 0.01
    step_decay::Float64 = 0.99
end

function initial_state(optimizer::Descent)
    optimizer.step_size
end

#====================================== optimization problem ======================================#

struct PoseTransformation{T} <: FieldVector{3,T}
    x::T
    y::T
    α::T
end

function center_of_mass(lines::AbstractVector{Line{N,T}}) where {N,T}
    total_com, total_mass = reduce(lines; init = (zero(Point{N,T}), 0.0)) do (cum_com, cum_mass), l
        p1, p2 = l
        line_vec = p2 - p1
        mass = norm(line_vec)
        com = p1 + 1 // 2 * line_vec
        new_cum_mass = cum_mass + mass
        @assert new_cum_mass > 0
        new_com = (cum_com * cum_mass + com * mass) / new_cum_mass
        (new_com, new_cum_mass)
    end
    total_com, total_mass
end

function fit_line_transformation(
    lines,
    map_lines;
    n_iterations_max = 50,
    snapshot_stepsize = show_debug_animation ? 1 : nothing,
    min_grad_norm = 1e-3,
    min_cost = 0.1,
    optimizer = LevenBergMarquardt(),
)
    θ::PoseTransformation{Float64} = zero(PoseTransformation)
    ∇θ::PoseTransformation{Float64} = zero(PoseTransformation)

    "Normalize transformation to rotate around com of lines."
    lines_com, lines_mass = center_of_mass(lines)
    debug_snapshots = []

    optimizer_state = initial_state(optimizer)
    cost_cache = Inf

    function cost(params)
        tform = pose_transformation(params; rot_center = lines_com)
        c = sum(l -> line_fit_error(transform(tform, l), map_lines), lines)
        cost_cache = ForwardDiff.value(c)
        c
    end

    converged = false

    for i in 1:n_iterations_max
        ∇θ = ForwardDiff.gradient(cost, θ)
        ∇θ_norm = norm(∇θ)
        if cost_cache < sqrt(lines_mass) * min_cost
            converged = true
            break
        end
        if ∇θ_norm < min_grad_norm
            break
        end

        θ, optimizer_state = first_order_step(optimizer, cost, cost_cache, θ, ∇θ, optimizer_state)

        # take a snapshot every few iterations
        if !isnothing(snapshot_stepsize) && iszero(i % snapshot_stepsize)
            tform_snapshot = pose_transformation(θ)
            cost_snapshot = cost(θ)
            push!(debug_snapshots, (; i, tform_snapshot, cost_snapshot))
        end
    end

    pose_transformation(θ), converged, debug_snapshots
end

#============================================ test run ============================================#

"Artificial noise on lines."
function noisify(line; max_segments = 3, rng = Random.GLOBAL_RNG, σx=0.01, σy=0.01)
    n_segments = rand(1:max_segments)
    line_vec, p1, p2 = line_vector_representation(line)
    line_segement_thresholds = vcat(0.0, sort(rand(rng, n_segments - 1)), 1.0)
    rand_translation(rng) = Translation(σx * randn(rng), σy * randn(rng))
    map(line_segement_thresholds, vcat(line_segement_thresholds[2:end])) do t_start, t_end
        p_start = rand_translation(rng)(p1 + t_start * line_vec)
        p_end = rand_translation(rng)(p1 + t_end * line_vec)
        Line(p_start, p_end)
    end
end

"The known lines on the map."
spl_field = SPLField()
"Some perceived lines."
noise_tform = pose_transformation(randn(3))
initial_lines =
    map([Line(Point(0.0, 1.0), Point(2.0, 1.0)), Line(Point(1.0, -1.0), Point(1.0, 1.0))]) do l
        transform(noise_tform, l)
    end |> lines -> mapreduce(noisify, vcat, lines)

fitted_line_transformation, converged, snapshots =
    @time fit_line_transformation(initial_lines, spl_field.lines)
transformed_lines = map(l -> transform(fitted_line_transformation, l), initial_lines)

@show converged

static_line_data = vcat(
    line_dataframe(spl_field.lines, "map"),
    line_dataframe(initial_lines, "initial"),
    line_dataframe(transformed_lines, "final_transformation"),
)
visualize(static_line_data) |> display

function debug_viz()
    dynamic_line_data = mapreduce(vcat, snapshots) do (i, tform_snapshot)
        transformed_lines = map(l -> transform(tform_snapshot, l), initial_lines)
        line_dataframe(transformed_lines, "optimization step", i)
    end

    animate(
        visualize(static_line_data) +
        geom_segment(data = dynamic_line_data) +
        transition_manual(:i_iter),
        nframes = nrow(dynamic_line_data),
        width = 1000,
        height = 500,
        units = "px",
    )
end

if show_debug_animation && converged
    debug_viz()
end
