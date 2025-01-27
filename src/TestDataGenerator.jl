module TestDataGenerator

using CoordinateTransformations: Translation
using GeometryBasics: Line
using Random: GLOBAL_RNG, shuffle

# include("SPLFieldModel.jl")
import ..SPLFieldModel

# include("GeometryTransformationUtils.jl")
using ..GeometryTransformationUtils: direction_vector, pose_transformation

"Artificial noise on lines."
function noisify(line; max_n_segments = 3, rng = Random.GLOBAL_RNG, σx = 0.01, σy = 0.01)
    n_segments = rand(1:max_n_segments)
    line_vec, p1, p2 = direction_vector(line)
    line_segement_thresholds = vcat(0.0, sort(rand(rng, n_segments - 1)), 1.0)
    rand_translation() = Translation(σx * randn(rng), σy * randn(rng))

    map(line_segement_thresholds, vcat(line_segement_thresholds[2:end])) do t_start, t_end
        p_start = rand_translation()(p1 + t_start * line_vec)
        p_end = rand_translation()(p1 + t_end * line_vec)
        Line(p_start, p_end)
    end
end

"""
Sample up to `max_n_lines` from `spl_field.lines` (without replacement) and apply a random
`rigid_transformation` and a artificial noise to each line.
"""
function generate_lines(
    n_lines = 2:5,
    pose_noise = (1.0, 1.0, deg2rad(15)),
    spl_field = SPLFieldModel.SPLField(),
    rng = GLOBAL_RNG,
    kwargs...,
)
    lines = Iterators.take(shuffle(rng, spl_field.lines), rand(rng, n_lines))
    rigid_tform = let
        x, y, α = 2 * (rand(rng, 3) .- 0.5) .* pose_noise
        pose_transformation(x, y, α)
    end
    mapreduce(vcat, lines) do l
        noisify(rigid_tform(l); rng, kwargs...)
    end
end

end # module
