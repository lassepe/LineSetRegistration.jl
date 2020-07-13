using CoordinateTransformations: Translation, LinearMap
using DataFrames: DataFrame, nrow
using GeometryBasics: Point, Line
using Lazy: @forward
using LinearAlgebra: diagm, norm, normalize
using RCall
using StaticArrays: FieldVector, SMatrix, SVector, SizedVector
using VegaLite
using ForwardDiff
using LinearAlgebra: ⋅
using Query

@rlibrary ggplot2
@rlibrary gganimate

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

function distance_segment_sq(point, line)
    p_start, p_end = line
    line_vector = p_end - p_start
    line_len_sq = line_vector ⋅ line_vector
    @assert line_len_sq > 0
    t = ((point - p_start) ⋅ line_vector) / line_len_sq

    closest_point_on_line = if t < 0
        p_start
    elseif t > 1
        p_end
    else
        p_start + line_vector .* t
    end

    dp = point - closest_point_on_line
    dp ⋅ dp
end

function distance_segment(point, line)
    sqrt(distance_segment_sq(point, line))
end

function line_length_sq(line::Line)
    p1, p2 = line
    line_vec = p2 - p1
    line_vec ⋅ line_vec
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

function pose_transformation(transformation_parameters)
    pose_transformation(
        transformation_parameters[1],
        transformation_parameters[2],
        transformation_parameters[3],
    )
end

function pose_transformation(x, y, α)
    sα, cα = sin(α), cos(α)
    Translation(x, y) ∘ LinearMap(SMatrix{2,2}(cα, sα, -sα, cα))
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
    total_com
end

function fit_line_transformation(
    lines,
    map_lines;
    n_iterations_max = 50,
    step_size = 0.03,
    decay = 0.99,
    snapshot_stepsize = 1,
    min_grad_norm = 1e-3,
    min_cost = 0.2,
)
    θ::PoseTransformation{Float64} = zero(PoseTransformation)
    ∇θ::PoseTransformation{Float64} = zero(PoseTransformation)

    "Normalize transformation to rotate around com of lines."
    lines_com_tform = Translation(center_of_mass(lines))
    # TODO: this does not really seem to be neccessary
    normalized_tform(params) = lines_com_tform ∘ pose_transformation(params) ∘ inv(lines_com_tform)
    debug_snapshots = []

    cost_cache = Inf

    function cost(params)
        tform = normalized_tform(params)
        c = sum(l -> line_fit_error(transform(tform, l), map_lines), lines)
        cost_cache = ForwardDiff.value(c)
        c
    end

    converged = false

    for i in 1:n_iterations_max
        ∇θ = ForwardDiff.gradient(cost, θ)
        ∇θ_norm = norm(∇θ)
        if cost_cache < min_cost
            converged = true
            break
        end
        if ∇θ_norm < min_grad_norm
            break
        end
        θ -= step_size * ∇θ #/ ∇θ_norm
        step_size *= decay

        # take a snapshot every few iterations
        if !isnothing(snapshot_stepsize) && iszero(i % snapshot_stepsize)
            tform_snapshot = normalized_tform(θ)
            cost_snapshot = cost(θ)
            push!(debug_snapshots, (; i, tform_snapshot, cost_snapshot))
        end
    end

    normalized_tform(θ), converged, debug_snapshots
end

#============================================ test run ============================================#

"The known lines on the map."
spl_field = SPLField()
"Some perceived lines."
noise_tform = pose_transformation(randn(), randn(), randn())
initial_lines =
    map([Line(Point(1.0, 1.0), Point(2.0, 1.0)), Line(Point(1.0, -1.0), Point(1.0, 1.0))]) do l
        l
    end |> lines -> map(l -> transform(noise_tform, l), lines)

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
    )
end

if converged
    debug_viz()
end
